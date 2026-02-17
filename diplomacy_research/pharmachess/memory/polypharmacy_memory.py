"""Polypharmacy Memory Bank — Richelieu-Style Self-Evolving Memory

Directly mirrors the memory mechanism described in the Richelieu paper
(NeurIPS 2024, §3.3):

  "Richelieu augments memories by self-play games for self-evolving
   without any human annotation … experiences with high evaluative scores
   reinforce successful strategies."

In PharmaChess:
  - "Experiences" = evaluated polypharmacy regimens with FAERS-derived
    outcome scores.
  - "High-scoring" = low adverse event rate after intervention.
  - Similarity retrieval = Jaccard similarity over shared CYP pathways
    (analogous to Richelieu's state-based similarity function).
  - Self-evolution = system is run on FAERS temporal data; validated
    against known signals without human labelling.

The memory bank is persistence-capable (JSON lines) so that it grows
across multiple episodes (self-play iterations).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single stored polypharmacy evaluation.

    Analogous to a "game history" stored in Richelieu's memory bank.

    Attributes:
        entry_id:        SHA-256 hash of regimen fingerprint + timestamp.
        regimen:         Sorted list of drug names (canonical form).
        cyp_pathways:    Set of metabolic nodes occupied by this regimen.
        risk_score:      Overall risk score at episode end [0, 1].
        faers_adr_rate:  FAERS adverse event rate for this regimen.
        conflict_score:  CYP graph conflict score.
        agent_actions:   Final actions proposed by drug agents.
        outcome_label:   "safe" | "moderate" | "high_risk" | "contraindicated"
        explanation:     LLM-generated reasoning summary.
        timestamp:       UNIX timestamp of creation.
        eval_score:      Composite evaluation score used for retrieval ranking.
                         Higher = more informative (high-risk OR confirmed-safe).
        source:          "self_play" | "faers_historical" | "clinical_review"
    """
    entry_id:       str
    regimen:        List[str]
    cyp_pathways:   List[str]   # sorted list for JSON serialisability
    risk_score:     float
    faers_adr_rate: float
    conflict_score: float
    agent_actions:  Dict[str, str]
    outcome_label:  str
    explanation:    str
    timestamp:      float
    eval_score:     float
    source:         str = "self_play"

    @classmethod
    def create(cls, regimen: List[str], cyp_pathways: Set[str],
                risk_score: float, faers_adr_rate: float,
                conflict_score: float, agent_actions: Dict[str, str],
                explanation: str, source: str = "self_play") -> "MemoryEntry":
        """Factory with automatic ID, label and eval_score generation."""
        sorted_regimen = sorted(r.lower() for r in regimen)
        sorted_pathways = sorted(cyp_pathways)
        entry_id = _make_entry_id(sorted_regimen)

        outcome_label = _classify_outcome(risk_score)

        # Eval score: entries are most valuable when they are extreme (very
        # safe or very risky) — similar to Richelieu's "high evaluative score"
        eval_score = abs(risk_score - 0.5) * 2  # peaks at 0 and 1

        return cls(
            entry_id=entry_id,
            regimen=sorted_regimen,
            cyp_pathways=sorted_pathways,
            risk_score=round(risk_score, 4),
            faers_adr_rate=round(faers_adr_rate, 4),
            conflict_score=round(conflict_score, 4),
            agent_actions=agent_actions,
            outcome_label=outcome_label,
            explanation=explanation,
            timestamp=time.time(),
            eval_score=round(eval_score, 4),
            source=source,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(**data)


# ---------------------------------------------------------------------------
# Memory bank
# ---------------------------------------------------------------------------

class PolypharmacyMemory:
    """Growing repository of polypharmacy evaluations.

    Supports:
      - add(entry)          : Store a new evaluation
      - retrieve_similar()  : Jaccard-similarity retrieval (Richelieu §3.3)
      - save() / load()     : Persistence to JSON-lines file
      - self_play_update()  : Batch-update from episode logs (self-evolution)

    Thread-safety: single-threaded async use only (no locking).
    """

    def __init__(self, persist_path: Optional[str] = None,
                 max_size: int = 10_000):
        """
        Args:
            persist_path: If provided, memory is loaded from / saved to this
                          JSON-lines file.  Enables cross-episode learning.
            max_size:     Maximum number of entries.  Oldest low-eval-score
                          entries are evicted when the cap is reached.
        """
        self.persist_path = Path(persist_path) if persist_path else None
        self.max_size = max_size
        self._entries: List[MemoryEntry] = []
        self._id_set: Set[str] = set()

        if self.persist_path and self.persist_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> bool:
        """Add a new entry to the memory bank.

        Returns:
            True if entry was added, False if a duplicate ID already exists.
        """
        if entry.entry_id in self._id_set:
            return False

        self._entries.append(entry)
        self._id_set.add(entry.entry_id)

        # Evict if over capacity (remove lowest eval_score entries)
        if len(self._entries) > self.max_size:
            self._entries.sort(key=lambda e: e.eval_score, reverse=True)
            evicted = self._entries[self.max_size:]
            self._entries = self._entries[:self.max_size]
            for ev in evicted:
                self._id_set.discard(ev.entry_id)

        return True

    def retrieve_similar(
        self,
        drugs: List[str],
        cyp_profile: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        min_similarity: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most similar past evaluations for a drug regimen.

        Similarity metric (Richelieu-style):
          Primary:   Jaccard similarity on CYP pathway sets
          Secondary: Jaccard similarity on drug name sets
          Combined:  0.6 * cyp_sim + 0.4 * drug_sim

        This mirrors Richelieu's "similarity-based function … to assess past
        interactions relevant to the current state" (paper §3.2).

        Args:
            drugs:          Current regimen drugs.
            cyp_profile:    CYP occupancy dict (optional; improves matching).
            top_k:          Number of results to return.
            min_similarity: Minimum combined similarity threshold.

        Returns:
            List of dicts, each containing entry fields + "similarity_score".
        """
        query_drugs = frozenset(d.lower() for d in drugs)

        # Extract CYP pathways from the occupancy dict
        if cyp_profile:
            query_pathways: Set[str] = set()
            for node, roles in cyp_profile.items():
                if any(roles.get(r) for r in ("substrates", "inhibitors", "inducers")):
                    query_pathways.add(node)
        else:
            query_pathways = set()

        scored: List[Tuple[float, MemoryEntry]] = []
        for entry in self._entries:
            entry_drugs = frozenset(entry.regimen)
            drug_sim = _jaccard(query_drugs, entry_drugs)

            if query_pathways:
                entry_pathways = frozenset(entry.cyp_pathways)
                cyp_sim = _jaccard(query_pathways, entry_pathways)
                combined = 0.6 * cyp_sim + 0.4 * drug_sim
            else:
                combined = drug_sim

            if combined >= min_similarity:
                scored.append((combined, entry))

        scored.sort(key=lambda x: (-x[0], -x[1].eval_score))
        results = []
        for sim, entry in scored[:top_k]:
            d = entry.to_dict()
            d["similarity_score"] = round(sim, 3)
            results.append(d)
        return results

    def self_play_update(self, episode_logs: List[List[Dict]],
                          explanations: Optional[List[str]] = None) -> int:
        """Batch-update the memory bank from self-play episode logs.

        This is the core of Richelieu's self-evolution mechanism:
          "Multi-agent self-play games are employed — the agents control all
           countries to simulate and acquire diverse experiences." (§3.4)

        In PharmaChess the 'simulation' is running the environment on
        diverse drug combinations (drawn from FAERS or randomly generated)
        and storing the outcomes.

        Args:
            episode_logs: List of episode logs from PolypharmacyEnv.get_episode_log().
                          Each log is a list of state dicts across turns.
            explanations: Optional per-episode LLM explanation strings.

        Returns:
            Number of new entries added to the memory bank.
        """
        added = 0
        for idx, log in enumerate(episode_logs):
            if not log:
                continue
            final_state = log[-1]  # last turn = final outcome
            explanation = (explanations or [])[idx] if explanations else ""

            # Extract CYP pathways from cyp_occupancy (if present in log)
            cyp_pathways = set()
            # agent_assessments may contain cyp_conflicts per drug
            for drug_assessment in final_state.get("agent_assessments", {}).values():
                cyp_pathways.update(drug_assessment.get("cyp_conflicts", []))

            entry = MemoryEntry.create(
                regimen=final_state.get("drugs_in_regimen", []),
                cyp_pathways=cyp_pathways,
                risk_score=final_state.get("overall_risk_score", 0.5),
                faers_adr_rate=final_state.get("faers_adr_rate", 0.0),
                conflict_score=final_state.get("conflict_score", 0.0),
                agent_actions=final_state.get("agent_actions", {}),
                explanation=explanation,
                source="self_play",
            )
            if self.add(entry):
                added += 1

        return added

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist memory bank to a JSON-lines file."""
        target = Path(path) if path else self.persist_path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            for entry in self._entries:
                fh.write(json.dumps(entry.to_dict()) + "\n")

    def load(self, path: Optional[str] = None) -> int:
        """Load memory bank from a JSON-lines file.

        Returns:
            Number of entries loaded.
        """
        target = Path(path) if path else self.persist_path
        if target is None or not target.exists():
            return 0
        loaded = 0
        with open(target, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = MemoryEntry.from_dict(json.loads(line))
                    if entry.entry_id not in self._id_set:
                        self._entries.append(entry)
                        self._id_set.add(entry.entry_id)
                        loaded += 1
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
        return loaded

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the memory bank."""
        if not self._entries:
            return {"size": 0}
        risk_scores = [e.risk_score for e in self._entries]
        labels = {}
        for e in self._entries:
            labels[e.outcome_label] = labels.get(e.outcome_label, 0) + 1
        return {
            "size":              len(self._entries),
            "avg_risk_score":    round(sum(risk_scores) / len(risk_scores), 3),
            "max_risk_score":    round(max(risk_scores), 3),
            "min_risk_score":    round(min(risk_scores), 3),
            "outcome_breakdown": labels,
            "sources":           {
                s: sum(1 for e in self._entries if e.source == s)
                for s in {"self_play", "faers_historical", "clinical_review"}
            },
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<PolypharmacyMemory size={len(self)}>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two frozensets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _make_entry_id(sorted_regimen: List[str]) -> str:
    """Deterministic ID from the regimen (same drugs → same base ID)."""
    key = "|".join(sorted_regimen)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _classify_outcome(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "contraindicated"
    if risk_score >= 0.50:
        return "high_risk"
    if risk_score >= 0.25:
        return "moderate"
    return "safe"
