"""OpenAI-Compatible Drug Agent

Concrete implementation of BaseDrugAgent using any OpenAI-compatible
endpoint (OpenAI, Together AI, Groq, vLLM, etc.).

Mirrors diplomacy_sample/agents/llm_agents/azure_agent.py but:
  - Uses langchain_openai.ChatOpenAI instead of AzureChatOpenAI
  - Accepts base_url so it works with any OpenAI-compatible server
  - Specialised prompts and parsers for pharmacological reasoning
"""
import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from .base_drug_agent import BaseDrugAgent, OpenAIConfig, DRUG_ACTIONS


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RISK_ASSESSMENT_TEMPLATE = """\
You are a clinical pharmacologist evaluating drug safety for a polypharmacy patient.

## Your drug
Generic name : {drug_name}
Known CYP roles:
  Substrates  : {cyp_substrates}   (enzymes that metabolise this drug)
  Inhibitors  : {cyp_inhibitors}   (enzymes this drug inhibits)
  Inducers    : {cyp_inducers}     (enzymes this drug induces)

## Full patient regimen
Drugs in regimen: {drugs_in_regimen}

## Metabolic territory map (CYP occupancy)
Each CYP enzyme lists which drugs in the regimen are competing for it.
{cyp_occupancy_table}

## Known drug-drug interactions involving {drug_name}
{interaction_pairs}

## Real-world FAERS signal
Adverse event co-report rate for this regimen: {faers_adr_rate:.2%}
Key FAERS signals: {faers_signals}

## Similar historical cases (from memory bank)
{memory_precedents}

## Task
Assess the interaction risk that {drug_name} introduces into this regimen.
Identify every CYP enzyme where {drug_name} competes with another drug
(territory conflict) and rate the overall risk.

Respond ONLY with valid JSON matching exactly this schema — no extra text:
{{
  "drug": "{drug_name}",
  "risk_level": "<LOW|MODERATE|HIGH|CRITICAL>",
  "risk_score": <float 0.0-1.0>,
  "reasoning": "<2-4 sentence pharmacological explanation>",
  "cyp_conflicts": ["<CYP3A4>", ...],
  "flagged_pairs": ["<co-drug name>", ...]
}}
"""

_ACTION_PROPOSAL_TEMPLATE = """\
You are a clinical pharmacologist deciding the best course of action for a drug.

## Risk assessment for {drug_name}
Risk level : {risk_level}
Risk score : {risk_score}
Reasoning  : {reasoning}
CYP conflicts : {cyp_conflicts}
Flagged co-drugs : {flagged_pairs}

## Current regimen
{drugs_in_regimen}

## Available actions
{action_list}

Choose the single most appropriate action for {drug_name} to reduce patient risk.
Consider: clinical necessity of the drug, severity of identified interactions,
availability of safer alternatives.

Respond ONLY with a JSON object — no extra text:
{{
  "action": "<one of the available actions>",
  "rationale": "<1-2 sentence justification>"
}}
"""


class OpenAIDrugAgent(BaseDrugAgent):
    """Drug agent backed by any OpenAI-compatible LLM.

    Usage example:
        config = OpenAIConfig(
            model_name="gpt-4o",
            api_key=os.environ["OPENAI_API_KEY"],
        )
        agent = OpenAIDrugAgent(drug_name="rivaroxaban", config=config)
        risk = await agent.assess_interaction_risk(state)
        action = await agent.propose_action(state, risk)
    """

    def __init__(self, drug_name: str, config: OpenAIConfig,
                 cyp_profile: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            drug_name:   Generic drug name (e.g. "rivaroxaban").
            config:      OpenAI-compatible LLM config.
            cyp_profile: Pre-loaded CYP data for this drug from DrugBankLoader.
                         Keys: "substrates", "inhibitors", "inducers".
                         If None the agent fetches it lazily on first use.
        """
        self.cyp_profile = cyp_profile or {"substrates": [], "inhibitors": [], "inducers": []}
        super().__init__(drug_name, config)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_llm(self) -> ChatOpenAI:
        kwargs: Dict[str, Any] = dict(
            model=self.config.model_name,
            openai_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            max_retries=3,
            timeout=60,
        )
        if self.config.base_url:
            kwargs["openai_api_base"] = self.config.base_url
        if self.config.extra_headers:
            kwargs["default_headers"] = self.config.extra_headers
        return ChatOpenAI(**kwargs)

    def _init_prompts(self):
        self._risk_prompt = ChatPromptTemplate.from_template(_RISK_ASSESSMENT_TEMPLATE)
        self._action_prompt = ChatPromptTemplate.from_template(_ACTION_PROPOSAL_TEMPLATE)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    async def assess_interaction_risk(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Call the LLM to produce a structured pharmacological risk assessment."""
        cyp_occupancy_table = self._format_cyp_table(state.get("cyp_occupancy", {}))
        interaction_pairs = self._format_interactions(state.get("interaction_pairs", []))
        memory_text = self._format_memory(state.get("memory_precedents", []))
        faers_signals = self._format_faers_signals(state.get("faers_top_signals", []))

        chain = LLMChain(llm=self.llm, prompt=self._risk_prompt)

        # Small delay to respect rate limits (mirrors azure_agent.py pattern)
        await asyncio.sleep(0.2)

        raw = await chain.arun(
            drug_name=self.drug_name,
            cyp_substrates=", ".join(self.cyp_profile.get("substrates", [])) or "none known",
            cyp_inhibitors=", ".join(self.cyp_profile.get("inhibitors", [])) or "none known",
            cyp_inducers=", ".join(self.cyp_profile.get("inducers", [])) or "none known",
            drugs_in_regimen=", ".join(state.get("drugs_in_regimen", [])),
            cyp_occupancy_table=cyp_occupancy_table,
            interaction_pairs=interaction_pairs,
            faers_adr_rate=state.get("faers_adr_rate", 0.0),
            faers_signals=faers_signals,
            memory_precedents=memory_text,
        )

        assessment = self.parse_response(raw)
        self._record_assessment(assessment)
        return assessment

    async def propose_action(self, state: Dict[str, Any],
                             risk_assessment: Dict[str, Any]) -> str:
        """Given the risk assessment, propose the best action from DRUG_ACTIONS."""
        chain = LLMChain(llm=self.llm, prompt=self._action_prompt)
        await asyncio.sleep(0.2)

        raw = await chain.arun(
            drug_name=self.drug_name,
            risk_level=risk_assessment.get("risk_level", "UNKNOWN"),
            risk_score=risk_assessment.get("risk_score", 0.0),
            reasoning=risk_assessment.get("reasoning", ""),
            cyp_conflicts=", ".join(risk_assessment.get("cyp_conflicts", [])) or "none",
            flagged_pairs=", ".join(risk_assessment.get("flagged_pairs", [])) or "none",
            drugs_in_regimen=", ".join(state.get("drugs_in_regimen", [])),
            action_list="\n".join(f"  - {a}" for a in DRUG_ACTIONS),
        )

        try:
            parsed = json.loads(self._extract_json(raw))
            action = parsed.get("action", "HOLD").upper()
            if action not in DRUG_ACTIONS:
                action = "HOLD"
            return action
        except (json.JSONDecodeError, ValueError):
            return "HOLD"

    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse the LLM's JSON risk assessment, falling back gracefully."""
        try:
            data = json.loads(self._extract_json(raw_text))
            # Ensure required keys are present
            data.setdefault("drug", self.drug_name)
            data.setdefault("risk_level", "UNKNOWN")
            data.setdefault("risk_score", 0.5)
            data.setdefault("reasoning", raw_text[:300])
            data.setdefault("cyp_conflicts", [])
            data.setdefault("flagged_pairs", [])
            return data
        except (json.JSONDecodeError, ValueError):
            return {
                "drug": self.drug_name,
                "risk_level": "UNKNOWN",
                "risk_score": 0.5,
                "reasoning": raw_text[:300],
                "cyp_conflicts": [],
                "flagged_pairs": [],
                "parse_error": True,
            }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract the first JSON object or array from a text blob."""
        # Try to find a JSON block inside triple backticks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        # Fallback: find raw braces
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    @staticmethod
    def _format_cyp_table(cyp_occupancy: Dict[str, Any]) -> str:
        """Format CYP occupancy dict for LLM prompt.

        Handles the nested structure from DrugBankLoader.build_cyp_occupancy():
          {node_alias: {"substrates": [...], "inhibitors": [...], "inducers": [...]}}
        """
        if not cyp_occupancy:
            return "  (no occupancy data available)"
        lines = []
        for node, roles in sorted(cyp_occupancy.items()):
            if isinstance(roles, dict):
                # Nested structure from DrugBankLoader
                parts = []
                for role in ("substrates", "inhibitors", "inducers"):
                    drug_list = roles.get(role, [])
                    if drug_list:
                        parts.append(f"{role}: {', '.join(drug_list)}")
                if parts:
                    lines.append(f"  {node} | " + " | ".join(parts))
            else:
                # Flat list fallback
                lines.append(f"  {node}: {', '.join(roles) if roles else 'uncontested'}")
        return "\n".join(lines) if lines else "  (no occupied nodes)"

    @staticmethod
    def _format_interactions(pairs: List[Dict]) -> str:
        if not pairs:
            return "  (no known interactions found in DrugBank)"
        lines = []
        for p in pairs[:10]:  # cap at 10 to stay within context
            severity = p.get("severity", "unknown")
            description = p.get("description", "")[:120]
            lines.append(f"  [{severity.upper()}] {p.get('drug_a')} × {p.get('drug_b')}: {description}")
        return "\n".join(lines)

    @staticmethod
    def _format_memory(precedents: List[Dict]) -> str:
        if not precedents:
            return "  (no similar cases in memory bank yet — this may be a novel combination)"
        lines = []
        for i, p in enumerate(precedents[:5], 1):
            # MemoryEntry field is "outcome_label", not "outcome"
            outcome = p.get("outcome_label", p.get("outcome", "unknown"))
            regimen = ", ".join(p.get("regimen", []))
            score = p.get("similarity_score", 0.0)
            explanation = p.get("explanation", "")[:120]
            lines.append(
                f"  [{i}] Regimen: {regimen}\n"
                f"      Outcome: {outcome} | Memory similarity: {score:.2f}\n"
                f"      Lesson: {explanation}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_faers_signals(signals: List[Dict]) -> str:
        if not signals:
            return "none retrieved"
        parts = []
        for s in signals[:5]:
            term = s.get("term", "")
            count = s.get("count", 0)
            parts.append(f"{term}({count})")
        return ", ".join(parts)
