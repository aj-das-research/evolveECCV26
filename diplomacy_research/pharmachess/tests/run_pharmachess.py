"""PharmaChess: Main Orchestration Script

Mirrors diplomacy_sample/tests/azure_vs_rule.py but for polypharmacy risk.

This script:
  1. Loads DrugBank data (dhimmel TSVs or built-in fallback profiles).
  2. Initialises one OpenAIDrugAgent per drug in the patient regimen.
  3. Initialises a RegimenCoordinator (the Richelieu-equivalent master agent).
  4. Runs a full evaluation episode via PolypharmacyEnv.
  5. Stores the episode in the memory bank (self-play update).
  6. Optionally runs a self-play loop over multiple regimens.

Environment variables required:
  OPENAI_API_KEY          — OpenAI API key (or compatible endpoint key)
  OPENAI_BASE_URL         — (optional) Override for compatible APIs
  OPENAI_MODEL            — Model name, default "gpt-4o"
  OPENFDA_API_KEY         — (optional) 120K req/day instead of 240/min
  NCBI_API_KEY            — (optional) PubMed 10 req/s instead of 3
  DRUGBANK_DATA_DIR       — (optional) Path to dhimmel/drugbank data/ dir
  PHARMACHESS_MEMORY_PATH — (optional) Path to persist the memory bank

Example usage:
    # Single regimen evaluation
    OPENAI_API_KEY=sk-... python run_pharmachess.py

    # Self-play loop over 20 random regimens
    OPENAI_API_KEY=sk-... python run_pharmachess.py --self-play 20

    # Custom regimen
    OPENAI_API_KEY=sk-... python run_pharmachess.py \
        --drugs rivaroxaban aspirin atorvastatin clarithromycin
"""
import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── ensure the project root is in sys.path ──────────────────────────────────
# File is at: pharmachess/tests/run_pharmachess.py
# parents[0]=tests, [1]=pharmachess, [2]=diplomacy_research, [3]=project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from diplomacy_research.pharmachess.agents.base_drug_agent import OpenAIConfig
from diplomacy_research.pharmachess.agents.openai_drug_agent import OpenAIDrugAgent
from diplomacy_research.pharmachess.data.faers_client import FAERSClient
from diplomacy_research.pharmachess.data.drugbank_loader import DrugBankLoader, _BUILTIN_CYP_PROFILES
from diplomacy_research.pharmachess.data.pubmed_client import PubMedClient
from diplomacy_research.pharmachess.env.polypharmacy_env import PolypharmacyEnv, PolypharmacyState
from diplomacy_research.pharmachess.graph.cyp_graph import build_graph_for_regimen
from diplomacy_research.pharmachess.memory.polypharmacy_memory import PolypharmacyMemory


# ---------------------------------------------------------------------------
# Default test regimens
# ---------------------------------------------------------------------------

DEFAULT_REGIMEN = [
    "rivaroxaban",      # CYP3A4 substrate + P-gp substrate
    "aspirin",          # antiplatelet — bleeding risk amplifier
    "atorvastatin",     # CYP3A4 substrate
    "clarithromycin",   # Strong CYP3A4 + P-gp inhibitor ← HIGH CONFLICT
]

# Regimen pool for self-play (diverse clinical scenarios)
SELF_PLAY_REGIMENS = [
    ["warfarin", "fluconazole", "omeprazole"],              # CYP2C9 conflict
    ["clopidogrel", "omeprazole", "aspirin"],               # CYP2C19 conflict
    ["metoprolol", "fluoxetine", "paroxetine"],             # CYP2D6 conflict
    ["simvastatin", "clarithromycin", "amlodipine"],        # CYP3A4 conflict
    ["rivaroxaban", "rifampicin"],                          # P-gp + CYP3A4 induction
    ["apixaban", "aspirin", "metformin"],                   # relatively safe
    ["warfarin", "aspirin", "omeprazole", "amiodarone"],    # classic high-risk
    ["phenytoin", "warfarin", "fluconazole"],               # CYP2C9 induction + inhibition
    ["carbamazepine", "atorvastatin", "metoprolol"],        # CYP3A4/2D6 induction
    ["dabigatran", "clarithromycin", "aspirin"],            # P-gp inhibition
    ["metformin", "amlodipine", "rosuvastatin"],            # low interaction
    ["rivaroxaban", "aspirin", "atorvastatin", "clarithromycin"],   # DEFAULT
]


# ---------------------------------------------------------------------------
# Coordinator (Richelieu-equivalent master agent)
# ---------------------------------------------------------------------------

class RegimenCoordinator:
    """The PharmaChess equivalent of Richelieu's main agent.

    Receives individual drug agent reports and synthesises a final verdict,
    analogous to Richelieu's 'Goal Planning with Reflection' module.
    """

    def __init__(self, config: OpenAIConfig, prompt_path: Optional[str] = None):
        self.config = config
        llm_kwargs: Dict[str, Any] = dict(
            model=config.model_name,
            openai_api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        if config.base_url:
            llm_kwargs["openai_api_base"] = config.base_url
        self.llm = ChatOpenAI(**llm_kwargs)

        # Load coordinator prompt (drug_interaction_prompt.txt)
        if prompt_path is None:
            prompt_path = Path(__file__).parent.parent / "prompts" / "drug_interaction_prompt.txt"
        with open(prompt_path, encoding="utf-8") as fh:
            self._template = fh.read()

    async def synthesise(self, state: PolypharmacyState) -> Dict[str, Any]:
        """Synthesise a final verdict from all agent assessments."""
        prompt = ChatPromptTemplate.from_template(self._template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        await asyncio.sleep(0.3)  # rate-limit courtesy

        raw = await chain.arun(
            drugs_in_regimen=", ".join(state.drugs_in_regimen),
            n_drugs=len(state.drugs_in_regimen),
            adjacency_table=_format_adjacency(state.graph),
            cyp_occupancy_table=_format_occupancy(state.cyp_occupancy),
            conflict_list=_format_conflicts(state.graph),
            supply_pressure_table=_format_pressure(state.supply_pressure),
            faers_adr_rate=state.faers_adr_rate,
            faers_signals=_format_reactions(state.faers_top_signals),
            faers_stats="(see pairwise FAERS query results above)",
            agent_reports=_format_agent_reports(state.agent_assessments),
            memory_precedents=_format_memory(state.memory_precedents),
            current_turn=state.turn,
            max_turns=PolypharmacyEnv.MAX_TURNS,
        )

        return _parse_json_response(raw, fallback={
            "overall_risk_level": "UNKNOWN",
            "overall_risk_score": state.overall_risk_score,
            "primary_concern": "Parse error — see raw LLM response",
            "coordinator_reasoning": raw[:400],
        })


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

async def run_episode(
    drugs: List[str],
    config: OpenAIConfig,
    faers_client: FAERSClient,
    db_loader: DrugBankLoader,
    pubmed_client: Optional[PubMedClient],
    memory: PolypharmacyMemory,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one complete PharmaChess evaluation episode.

    Analogous to run_test_game() in azure_vs_rule.py.

    Steps:
      1. Initialise one OpenAIDrugAgent per drug.
      2. Initialise RegimenCoordinator.
      3. Reset PolypharmacyEnv → get initial state.
      4. Each agent assesses its drug → all propose actions.
      5. Coordinator synthesises → produces final verdict.
      6. env.step() applies actions → new state.
      7. Repeat until done.
      8. Store episode in memory bank.
    """
    if verbose:
        print(f"\n{'═'*60}")
        print(f"PHARMACHESS EPISODE")
        print(f"Regimen: {', '.join(drugs)}")
        print(f"{'═'*60}")

    # Build one drug agent per drug in the regimen (mirrors 7 powers in Diplomacy)
    drug_agents: Dict[str, OpenAIDrugAgent] = {}
    for drug in drugs:
        cyp_profile = db_loader.get_cyp_profile(drug)
        drug_agents[drug] = OpenAIDrugAgent(
            drug_name=drug,
            config=config,
            cyp_profile=cyp_profile,
        )

    coordinator = RegimenCoordinator(config)

    env = PolypharmacyEnv(
        drugs=drugs,
        faers_client=faers_client,
        drugbank_loader=db_loader,
        pubmed_client=pubmed_client,
        memory=memory,
        verbose=verbose,
    )

    state = await env.reset()
    coordinator_reports: List[Dict] = []

    # ── Game loop ─────────────────────────────────────────────────────────
    while not state.is_done:
        if verbose:
            print(f"\n── Turn {state.turn + 1} ──")

        # Phase 1: Each drug agent assesses its risk (parallel — like all
        # powers submitting orders simultaneously in Diplomacy)
        assessment_tasks = {
            drug: agent.assess_interaction_risk(state.to_dict() if hasattr(state, 'to_dict') else _state_to_dict(state))
            for drug, agent in drug_agents.items()
            if drug in state.drugs_in_regimen
        }
        assessments = {}
        for drug, task in assessment_tasks.items():
            assessments[drug] = await task
            if verbose:
                print(
                    f"  [{drug}] Risk: {assessments[drug].get('risk_level', '?')} "
                    f"({assessments[drug].get('risk_score', 0):.2f}) | "
                    f"Conflicts: {assessments[drug].get('cyp_conflicts', [])}"
                )

        state.agent_assessments = assessments

        # Phase 2: Coordinator synthesises (Richelieu master-agent step)
        coordinator_verdict = await coordinator.synthesise(state)
        coordinator_reports.append(coordinator_verdict)
        if verbose:
            print(f"\n  COORDINATOR VERDICT: {coordinator_verdict.get('overall_risk_level', '?')} "
                  f"(score={coordinator_verdict.get('overall_risk_score', 0):.3f})")
            print(f"  Primary concern: {coordinator_verdict.get('primary_concern', '')}")

        # Phase 3: Each agent proposes an action
        state_dict = _state_to_dict(state)
        action_tasks = {
            drug: agent.propose_action(state_dict, assessments.get(drug, {}))
            for drug, agent in drug_agents.items()
            if drug in state.drugs_in_regimen
        }
        actions: Dict[str, str] = {}
        for drug, task in action_tasks.items():
            actions[drug] = await task

        if verbose:
            for drug, action in actions.items():
                if action != "HOLD":
                    print(f"  [{drug}] → {action}")

        # Phase 4: Environment step (applies actions, updates state)
        state, reward, info = await env.step(actions)

        if verbose:
            print(f"  Risk delta: {reward:+.4f} | Turn: {info['turn']}")

    # ── Episode complete ──────────────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*60}")
        print(f"EPISODE COMPLETE — Final state:")
        print(env.render())

    # Self-play memory update (Richelieu evolution step)
    final_verdict = coordinator_reports[-1] if coordinator_reports else {}
    n_added = memory.self_play_update(
        episode_logs=[env.get_episode_log()],
        explanations=[final_verdict.get("coordinator_reasoning", "")],
    )
    if verbose:
        print(f"Memory bank updated: {n_added} new entry/ies | Bank size: {len(memory)}")

    return {
        "regimen": drugs,
        "final_state": _state_to_dict(state),
        "final_verdict": final_verdict,
        "episode_turns": state.turn,
        "memory_entries_added": n_added,
    }


async def run_self_play(
    regimens: List[List[str]],
    config: OpenAIConfig,
    faers_client: FAERSClient,
    db_loader: DrugBankLoader,
    pubmed_client: Optional[PubMedClient],
    memory: PolypharmacyMemory,
    memory_save_path: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Run the self-play evolution loop over multiple regimens.

    Mirrors Richelieu's self-play mechanism where all countries are controlled
    by agents to generate diverse experiences.

    After all episodes, memory is saved (persistent across runs).
    """
    print(f"\n{'═'*60}")
    print(f"PHARMACHESS SELF-PLAY LOOP")
    print(f"Episodes: {len(regimens)} | Memory start size: {len(memory)}")
    print(f"{'═'*60}\n")

    results = []
    for i, drugs in enumerate(regimens):
        print(f"\n[Episode {i+1}/{len(regimens)}]")
        try:
            result = await run_episode(
                drugs=drugs,
                config=config,
                faers_client=faers_client,
                db_loader=db_loader,
                pubmed_client=pubmed_client,
                memory=memory,
                verbose=verbose,
            )
            results.append(result)
        except Exception as exc:
            print(f"  [ERROR] Episode failed: {exc}")
            continue

    print(f"\n{'═'*60}")
    print(f"SELF-PLAY COMPLETE")
    print(f"Episodes run: {len(results)} | Final memory size: {len(memory)}")
    stats = memory.stats()
    print(f"Memory stats: {json.dumps(stats, indent=2)}")

    if memory_save_path:
        memory.save(memory_save_path)
        print(f"Memory saved to: {memory_save_path}")

    return results


# ---------------------------------------------------------------------------
# Formatting helpers (analogous to azure_agent.py's format_* methods)
# ---------------------------------------------------------------------------

def _state_to_dict(state: PolypharmacyState) -> Dict[str, Any]:
    """Convert PolypharmacyState to a plain dict for agent consumption."""
    return {
        "drugs_in_regimen":   state.drugs_in_regimen,
        "cyp_occupancy":      state.cyp_occupancy,
        "interaction_pairs":  state.interaction_pairs,
        "faers_adr_rate":     state.faers_adr_rate,
        "faers_top_signals":  state.faers_top_signals,
        "memory_precedents":  state.memory_precedents,
        "conflict_score":     state.conflict_score,
        "supply_pressure":    state.supply_pressure,
        "current_turn":       state.turn,
        "overall_risk_score": state.overall_risk_score,
    }


def _format_adjacency(graph) -> str:
    if graph is None:
        return "  (graph not available)"
    adj = graph.to_adjacency_dict()
    lines = []
    for node, neighbours in adj.items():
        if neighbours:
            lines.append(f"  {node}: {', '.join(neighbours)}")
    return "\n".join(lines) if lines else "  (no adjacency data)"


def _format_occupancy(cyp_occupancy: Dict) -> str:
    if not cyp_occupancy:
        return "  (no occupancy data)"
    lines = []
    for node, roles in sorted(cyp_occupancy.items()):
        parts = []
        for role, drugs in roles.items():
            if drugs:
                parts.append(f"{role}: {', '.join(drugs)}")
        if parts:
            lines.append(f"  {node} | " + " | ".join(parts))
    return "\n".join(lines) if lines else "  (no occupied nodes)"


def _format_conflicts(graph) -> str:
    if graph is None or not graph.conflict_edges:
        return "  None detected"
    lines = []
    for a, b, node in sorted(graph.conflict_edges, key=lambda x: x[2]):
        lines.append(f"  [{node}] {a} inhibits → {b} substrate conflict")
    return "\n".join(lines[:10])  # cap at 10 lines


def _format_pressure(supply_pressure: Dict[str, int]) -> str:
    if not supply_pressure:
        return "  None"
    lines = []
    for node, count in sorted(supply_pressure.items(), key=lambda x: -x[1]):
        if count > 0:
            bar = "█" * count
            lines.append(f"  {node:12} {bar} ({count} drugs)")
    return "\n".join(lines) if lines else "  None"


def _format_reactions(signals: List[Dict]) -> str:
    if not signals:
        return "  (no FAERS signals available)"
    return "\n".join(
        f"  {s.get('term', '?')}: {s.get('count', 0)} reports"
        for s in signals[:8]
    )


def _format_agent_reports(assessments: Dict[str, Dict]) -> str:
    if not assessments:
        return "  (no agent reports yet)"
    lines = []
    for drug, report in assessments.items():
        rl = report.get("risk_level", "?")
        rs = report.get("risk_score", 0)
        conflicts = ", ".join(report.get("cyp_conflicts", [])) or "none"
        flagged = ", ".join(report.get("flagged_pairs", [])) or "none"
        reasoning = report.get("reasoning", "")[:200]
        lines.append(
            f"  ── {drug} ──\n"
            f"    Risk: {rl} ({rs:.2f})\n"
            f"    CYP conflicts: {conflicts}\n"
            f"    Flagged co-drugs: {flagged}\n"
            f"    Reasoning: {reasoning}"
        )
    return "\n".join(lines)


def _format_memory(precedents: List[Dict]) -> str:
    if not precedents:
        return "  (memory bank empty — first episode, no precedents)"
    lines = []
    for i, p in enumerate(precedents[:4], 1):
        regimen = ", ".join(p.get("regimen", []))
        outcome = p.get("outcome_label", "?")
        sim = p.get("similarity_score", 0)
        explanation = p.get("explanation", "")[:150]
        lines.append(
            f"  [{i}] Regimen: {regimen}\n"
            f"       Outcome: {outcome} | Similarity: {sim:.2f}\n"
            f"       Lesson: {explanation}"
        )
    return "\n".join(lines)


def _parse_json_response(raw: str, fallback: Dict) -> Dict:
    """Extract and parse a JSON object from an LLM response string."""
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        fallback["_raw"] = raw[:200]
        return fallback


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PharmaChess: Polypharmacy Risk Assessment via Multi-Agent LLM"
    )
    parser.add_argument(
        "--drugs", nargs="+", default=DEFAULT_REGIMEN,
        help="Generic drug names to evaluate (space-separated)"
    )
    parser.add_argument(
        "--self-play", type=int, default=0, metavar="N",
        help="Run self-play loop over N regimens from the built-in pool"
    )
    parser.add_argument(
        "--drugbank-dir", default=os.environ.get("DRUGBANK_DATA_DIR", ""),
        help="Path to dhimmel/drugbank data/ directory"
    )
    parser.add_argument(
        "--memory-path",
        default=os.environ.get("PHARMACHESS_MEMORY_PATH", "pharmachess_memory.jsonl"),
        help="Path to persist the memory bank"
    )
    parser.add_argument(
        "--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        help="OpenAI model name (or compatible)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress verbose output"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # ── Validate environment variables ────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    config = OpenAIConfig(
        model_name=args.model,
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=0.3,
        max_tokens=4000,
    )

    # ── Initialise data clients ───────────────────────────────────────────
    db_loader = DrugBankLoader(data_dir=args.drugbank_dir or "/tmp/drugbank")
    try:
        db_loader.load()
        print(f"DrugBank loaded from: {args.drugbank_dir}")
    except Exception:
        print("DrugBank TSVs not found — using built-in CYP profiles as fallback.")

    memory = PolypharmacyMemory(persist_path=args.memory_path)
    print(f"Memory bank initialised: {len(memory)} existing entries.")

    faers = FAERSClient(api_key=os.environ.get("OPENFDA_API_KEY"))
    pubmed = PubMedClient(api_key=os.environ.get("NCBI_API_KEY"))

    async with faers:  # manages aiohttp.ClientSession lifecycle
        if args.self_play > 0:
            # Draw from the built-in pool (or repeat if pool is smaller)
            regimens = (SELF_PLAY_REGIMENS * ((args.self_play // len(SELF_PLAY_REGIMENS)) + 1))[:args.self_play]
            random.shuffle(regimens)
            results = await run_self_play(
                regimens=regimens,
                config=config,
                faers_client=faers,
                db_loader=db_loader,
                pubmed_client=pubmed,
                memory=memory,
                memory_save_path=args.memory_path,
                verbose=not args.quiet,
            )
            print(f"\nSelf-play complete. {len(results)} episodes run.")
        else:
            result = await run_episode(
                drugs=args.drugs,
                config=config,
                faers_client=faers,
                db_loader=db_loader,
                pubmed_client=pubmed,
                memory=memory,
                verbose=not args.quiet,
            )
            print("\n── Final result ──")
            print(json.dumps({
                k: v for k, v in result.items()
                if k != "final_state"
            }, indent=2))

        # Always save memory after run
        memory.save(args.memory_path)


if __name__ == "__main__":
    asyncio.run(main())
