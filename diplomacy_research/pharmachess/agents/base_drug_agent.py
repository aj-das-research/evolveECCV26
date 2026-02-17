"""Base Drug Agent Module

Mirrors diplomacy_sample/agents/llm_agents/base_agent.py but for the
PharmaChess domain.  Each drug in a polypharmacy regimen is treated as an
autonomous agent that can assess its own metabolic risk and propose dosing
or substitution actions.

Analogy to Richelieu / Diplomacy:
  - Drug      ↔  Power (country)
  - CYP enzyme ↔  Supply centre (territory)
  - Inhibition ↔  Territory capture
  - Action     ↔  Order (hold / move / support)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class OpenAIConfig:
    """Configuration for an OpenAI-compatible LLM endpoint.

    Works with the public OpenAI API as well as any compatible server
    (Together AI, Groq, Ollama, vLLM, etc.) by setting base_url.
    """
    model_name: str                      # e.g. "gpt-4o", "gpt-4-turbo"
    api_key: str                         # OPENAI_API_KEY or equivalent
    base_url: Optional[str] = None       # Override for compatible APIs
    temperature: float = 0.3
    max_tokens: int = 4000
    extra_headers: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action vocabulary (mirrors Diplomacy order types)
# ---------------------------------------------------------------------------
DRUG_ACTIONS = [
    "HOLD",            # Keep current drug and dose — no change
    "INCREASE_DOSE",   # Dose escalation (may be countered by interactions)
    "DECREASE_DOSE",   # Dose reduction to reduce toxicity
    "SWITCH",          # Substitute with an alternative in the same class
    "DISCONTINUE",     # Remove drug from regimen entirely
    "FLAG_INTERACTION", # Raise an interaction alert for the coordinator
]


class BaseDrugAgent(ABC):
    """Abstract base class for a drug agent in the PharmaChess environment.

    One instance is created per drug in the polypharmacy regimen.  The agent
    receives the full regimen state (CYP occupancy, co-drug list, FAERS risk
    scores, memory-retrieved precedents) and outputs:
      1. A risk assessment (structured dict)
      2. A proposed action drawn from DRUG_ACTIONS
    """

    def __init__(self, drug_name: str, config: OpenAIConfig):
        """
        Args:
            drug_name: Generic name of the drug this agent represents.
            config:    LLM configuration.
        """
        self.drug_name = drug_name
        self.config = config

        self.llm = self._init_llm()
        self._init_prompts()

        # Running log of this agent's assessments across game turns
        self.assessment_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_llm(self):
        """Initialise and return the LLM client instance."""

    @abstractmethod
    def _init_prompts(self):
        """Load or build the prompt template(s) used by this agent."""

    @abstractmethod
    async def assess_interaction_risk(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the safety of this drug within the current regimen.

        Args:
            state: PolypharmacyState serialised as a plain dict containing:
                   - drugs_in_regimen: List[str]
                   - cyp_occupancy:    Dict[str, List[str]]  enzyme→drugs
                   - interaction_pairs: List[dict]           known DDIs
                   - faers_adr_rate:   float                 [0, 1]
                   - memory_precedents: List[dict]           similar past cases
                   - current_turn:     int

        Returns:
            {
              "drug": str,
              "risk_level": "LOW" | "MODERATE" | "HIGH" | "CRITICAL",
              "risk_score": float,   # 0–1
              "reasoning":  str,
              "cyp_conflicts": List[str],  # enzymes where conflict detected
              "flagged_pairs": List[str],  # co-drugs causing concern
            }
        """

    @abstractmethod
    async def propose_action(self, state: Dict[str, Any],
                             risk_assessment: Dict[str, Any]) -> str:
        """Propose the best action for this drug given the risk assessment.

        Returns one of the strings in DRUG_ACTIONS.
        """

    @abstractmethod
    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse raw LLM output into a structured assessment dict."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _record_assessment(self, assessment: Dict[str, Any]) -> None:
        """Append an assessment to this agent's internal history."""
        self.assessment_history.append(assessment)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the full assessment history for this drug agent."""
        return list(self.assessment_history)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} drug={self.drug_name!r}>"
