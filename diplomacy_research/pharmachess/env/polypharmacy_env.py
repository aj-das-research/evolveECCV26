"""PolypharmacyEnv — the PharmaChess Game Environment

Mirrors process.py / DiplomacyStrategyEnv but for polypharmacy risk assessment.

Diplomacy analogy:
  - Game loop turn     ↔  One cycle of all drug agents assessing the regimen
  - game.process()     ↔  env.step(actions) — applies agent decisions
  - supply centre count ↔  adverse event probability (reward signal)
  - game.is_game_done  ↔  env.is_done (converged / max turns reached)
  - game state proto   ↔  PolypharmacyState dataclass

The environment is also a data-collection harness for the self-play
(self-evolution) loop: it logs each full evaluation cycle so the memory
bank can be updated without requiring human-labelled ground truth.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..data.faers_client import FAERSClient
from ..data.drugbank_loader import DrugBankLoader
from ..data.pubmed_client import PubMedClient
from ..graph.cyp_graph import CYPGraph
from ..memory.polypharmacy_memory import PolypharmacyMemory


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

@dataclass
class PolypharmacyState:
    """Snapshot of the polypharmacy regimen at a given evaluation turn.

    Analogous to the SavedGame / State proto in diplomacy-research but as a
    plain Python dataclass so no Protobuf dependency is introduced.

    Attributes:
        drugs_in_regimen:   Current list of generic drug names.
        cyp_occupancy:      DrugBankLoader.build_cyp_occupancy() output.
        interaction_pairs:  List of known DDIs between drugs in the regimen.
        faers_adr_rate:     Adverse event co-report rate from FAERS [0, 1].
        faers_top_signals:  Top MedDRA reaction terms from FAERS.
        graph:              CYPGraph for this regimen.
        memory_precedents:  Similar past evaluations retrieved from memory bank.
        conflict_score:     Normalised graph conflict score [0, 1].
        supply_pressure:    Dict of key-node → competing-drug-count.
        turn:               Current evaluation turn (0-indexed).
        agent_assessments:  Dict of drug_name → latest risk assessment.
        agent_actions:      Dict of drug_name → latest proposed action.
        overall_risk_score: Aggregated risk score [0, 1] after coordinator.
        is_done:            Whether the evaluation has converged.
    """
    drugs_in_regimen:   List[str]             = field(default_factory=list)
    cyp_occupancy:      Dict[str, Any]        = field(default_factory=dict)
    interaction_pairs:  List[Dict]            = field(default_factory=list)
    faers_adr_rate:     float                 = 0.0
    faers_top_signals:  List[Dict]            = field(default_factory=list)
    graph:              Optional[CYPGraph]    = None
    memory_precedents:  List[Dict]            = field(default_factory=list)
    conflict_score:     float                 = 0.0
    supply_pressure:    Dict[str, int]        = field(default_factory=dict)
    turn:               int                   = 0
    agent_assessments:  Dict[str, Dict]       = field(default_factory=dict)
    agent_actions:      Dict[str, str]        = field(default_factory=dict)
    overall_risk_score: float                 = 0.0
    is_done:            bool                  = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise state for memory storage / logging (graph excluded)."""
        return {
            "drugs_in_regimen":  self.drugs_in_regimen,
            "faers_adr_rate":    self.faers_adr_rate,
            "conflict_score":    self.conflict_score,
            "supply_pressure":   self.supply_pressure,
            "overall_risk_score": self.overall_risk_score,
            "agent_assessments": self.agent_assessments,
            "agent_actions":     self.agent_actions,
            "turn":              self.turn,
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PolypharmacyEnv:
    """PharmaChess game environment.

    Manages one full evaluation episode for a polypharmacy regimen.
    An episode consists of:
      1. Initialisation: load DrugBank data, query FAERS, build CYP graph.
      2. Turns: each drug agent assesses its role and proposes an action.
         The coordinator synthesises the assessments.
      3. Step: apply actions (update regimen), recompute state, compute reward.
      4. Done: converge when risk stabilises or max_turns reached.

    Usage:
        env = PolypharmacyEnv(
            drugs=["rivaroxaban", "aspirin", "atorvastatin"],
            faers_client=FAERSClient(api_key=...),
            drugbank_loader=DrugBankLoader(data_dir=...),
        )
        state = await env.reset()
        while not state.is_done:
            state, reward, info = await env.step(agent_actions)
    """

    MAX_TURNS = 5       # Episode terminates after this many turns
    RISK_CONVERGE_DELTA = 0.02  # Episode terminates if risk changes < this

    def __init__(
        self,
        drugs: List[str],
        faers_client: FAERSClient,
        drugbank_loader: DrugBankLoader,
        pubmed_client: Optional[PubMedClient] = None,
        memory: Optional[PolypharmacyMemory] = None,
        verbose: bool = True,
    ):
        self.initial_drugs = list(drugs)
        self.faers = faers_client
        self.db = drugbank_loader
        self.pubmed = pubmed_client
        self.memory = memory
        self.verbose = verbose

        self._state: Optional[PolypharmacyState] = None
        self._prev_risk: float = 0.0
        self._episode_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Core gym-like interface
    # ------------------------------------------------------------------

    async def reset(self) -> PolypharmacyState:
        """Initialise a fresh evaluation episode and return the starting state.

        Analogous to game = Game() then the first game.process() in
        diplomacy_sample/tests/azure_vs_rule.py.
        """
        self._episode_log = []
        self._prev_risk = 0.0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PharmaChess Episode Start")
            print(f"Regimen: {', '.join(self.initial_drugs)}")
            print(f"{'='*60}")

        state = PolypharmacyState(drugs_in_regimen=list(self.initial_drugs))
        state = await self._populate_state(state)
        # Seed _prev_risk from the actual initial risk so the first step's
        # reward correctly measures improvement relative to the starting state.
        self._prev_risk = state.overall_risk_score
        self._state = state
        return state

    async def step(
        self, agent_actions: Dict[str, str]
    ) -> Tuple[PolypharmacyState, float, Dict[str, Any]]:
        """Apply agent actions, update the regimen, recompute state.

        Args:
            agent_actions: Dict of drug_name → action string from DRUG_ACTIONS.

        Returns:
            (new_state, reward, info_dict)
            reward is negative ADR probability (higher = safer regimen).
        """
        assert self._state is not None, "Call reset() before step()"
        state = self._state

        # Apply actions: modify the regimen
        new_drugs = self._apply_actions(state.drugs_in_regimen, agent_actions)

        if self.verbose:
            removed = set(state.drugs_in_regimen) - set(new_drugs)
            added = set(new_drugs) - set(state.drugs_in_regimen)
            if removed:
                print(f"  Actions removed: {', '.join(removed)}")
            if added:
                print(f"  Actions added  : {', '.join(added)}")

        # Build new state
        new_state = PolypharmacyState(
            drugs_in_regimen=new_drugs,
            turn=state.turn + 1,
            agent_actions=agent_actions,
        )
        new_state = await self._populate_state(new_state)

        # Compute reward: improvement in risk score (positive = improvement)
        reward = self._prev_risk - new_state.overall_risk_score
        self._prev_risk = new_state.overall_risk_score

        # Check termination
        new_state.is_done = (
            new_state.turn >= self.MAX_TURNS
            or abs(reward) < self.RISK_CONVERGE_DELTA
            or not new_state.drugs_in_regimen
        )

        # Log for self-play memory update
        self._episode_log.append(new_state.to_dict())

        info = {
            "turn": new_state.turn,
            "risk_delta": reward,
            "conflicts": len(new_state.graph.conflict_edges) if new_state.graph else 0,
        }

        self._state = new_state
        return new_state, reward, info

    def get_episode_log(self) -> List[Dict]:
        """Return the full sequence of states for this episode.

        Used by the self-play loop to populate the memory bank.
        """
        return list(self._episode_log)

    # ------------------------------------------------------------------
    # State population
    # ------------------------------------------------------------------

    async def _populate_state(self, state: PolypharmacyState) -> PolypharmacyState:
        """Fill in FAERS, DrugBank, graph, and memory data for a state."""
        drugs = state.drugs_in_regimen
        if not drugs:
            state.is_done = True
            return state

        # 1. DrugBank: CYP occupancy matrix + known DDIs
        state.cyp_occupancy = self.db.build_cyp_occupancy(drugs)
        state.interaction_pairs = self.db.get_all_interaction_pairs(drugs)

        # 2. Build CYP territorial conflict graph
        state.graph = CYPGraph.build(state.cyp_occupancy)
        state.conflict_score = state.graph.get_conflict_score(drugs)
        state.supply_pressure = state.graph.supply_centre_pressure(drugs)

        # 3. FAERS: adverse event co-report rate
        try:
            faers_result = await self.faers.query_regimen(drugs)
            state.faers_adr_rate = faers_result.get("adr_rate_estimate", 0.0)
            state.faers_top_signals = faers_result.get("top_reactions", [])
        except Exception as exc:
            if self.verbose:
                print(f"  [FAERS] Query failed: {exc}")
            state.faers_adr_rate = 0.0

        # 4. Memory bank: retrieve similar past evaluations (Richelieu-style)
        if self.memory:
            state.memory_precedents = self.memory.retrieve_similar(
                drugs, cyp_profile=state.cyp_occupancy, top_k=5
            )

        # 5. Overall risk score: weighted combination of evidence
        state.overall_risk_score = self._compute_overall_risk(state)

        if self.verbose:
            print(
                f"  Turn {state.turn} | Risk: {state.overall_risk_score:.3f} | "
                f"FAERS: {state.faers_adr_rate:.3f} | "
                f"Conflicts: {len(state.graph.conflict_edges) if state.graph else 0} | "
                f"DDIs: {len(state.interaction_pairs)}"
            )

        return state

    # ------------------------------------------------------------------
    # Risk aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_overall_risk(state: PolypharmacyState) -> float:
        """Aggregate risk from FAERS signal, DrugBank DDIs, and CYP conflicts.

        Weights reflect the Bradford Hill causality hierarchy:
          - FAERS statistical signal (real-world evidence)  : 0.40
          - Known DDI severity from DrugBank                : 0.35
          - CYP graph conflict score                        : 0.25

        Returns a score in [0, 1] where 1 = maximum risk.
        """
        # FAERS component (already normalised to [0, 1])
        faers_component = min(state.faers_adr_rate, 1.0)

        # DDI severity component
        severity_scores = {"contraindicated": 1.0, "major": 0.75,
                           "moderate": 0.45, "minor": 0.15}
        if state.interaction_pairs:
            ddi_scores = [
                severity_scores.get(p.get("severity", "minor"), 0.15)
                for p in state.interaction_pairs
            ]
            ddi_component = min(sum(ddi_scores) / len(ddi_scores) +
                                0.1 * (len(ddi_scores) - 1), 1.0)
        else:
            ddi_component = 0.0

        # Graph conflict component
        graph_component = state.conflict_score

        overall = (
            0.40 * faers_component +
            0.35 * ddi_component +
            0.25 * graph_component
        )
        return round(min(overall, 1.0), 4)

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_actions(drugs: List[str], actions: Dict[str, str]) -> List[str]:
        """Apply drug agent actions to produce the updated regimen.

        Actions:
          HOLD           — no change
          DECREASE_DOSE  — no structural change (tracked in log only)
          INCREASE_DOSE  — no structural change
          DISCONTINUE    — remove drug from regimen
          SWITCH         — remove drug (alternative not specified here; logged)
          FLAG_INTERACTION — no structural change; recorded for coordinator
        """
        remaining = list(drugs)
        for drug, action in actions.items():
            if action in ("DISCONTINUE", "SWITCH") and drug in remaining:
                remaining.remove(drug)
        return remaining

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> Optional[PolypharmacyState]:
        return self._state

    def render(self) -> str:
        """Return a human-readable string representation of the current state."""
        if self._state is None:
            return "<PolypharmacyEnv: not initialised>"
        s = self._state
        lines = [
            f"Turn {s.turn} | Drugs: {', '.join(s.drugs_in_regimen)}",
            f"Overall risk: {s.overall_risk_score:.3f}",
            f"FAERS ADR rate: {s.faers_adr_rate:.3f}",
            f"CYP conflicts: {len(s.graph.conflict_edges) if s.graph else 0}",
            f"Known DDIs: {len(s.interaction_pairs)}",
        ]
        if s.supply_pressure:
            top = sorted(s.supply_pressure.items(), key=lambda x: -x[1])[:3]
            lines.append("Supply pressure: " + ", ".join(f"{k}={v}" for k, v in top))
        return "\n".join(lines)
