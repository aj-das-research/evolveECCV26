"""DrugBank Loader

Loads pre-parsed DrugBank data from the dhimmel/drugbank GitHub repository
(https://github.com/dhimmel/drugbank — CC BY-NC 4.0).

The repository provides TSV files that are immediately usable without a
full DrugBank licence.  Key files consumed here:

  data/drugbank.tsv              — master drug table (ID, name, description, …)
  data/drug-interactions.tsv     — pairwise drug–drug interaction records
  data/drug-targets.tsv          — drug → target (gene) mappings
  data/drug-categories.tsv       — ATC / pharmacological category hierarchy

CYP enzyme occupancy data is synthesised from the target table by matching
known CYP gene symbols (CYP3A4, CYP2D6, …) and annotating each drug as
substrate / inhibitor / inducer where that information exists.

Usage:
    loader = DrugBankLoader(data_dir="/path/to/dhimmel-drugbank/data")
    loader.load()
    profile = loader.get_cyp_profile("rivaroxaban")
    interactions = loader.get_interactions("rivaroxaban")
"""
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# CYP enzymes and transporters that form the "game board" in PharmaChess.
# These map to "supply centres" in the Diplomacy analogy.
CYP_ENZYMES: Set[str] = {
    "CYP1A2", "CYP2B6", "CYP2C8", "CYP2C9", "CYP2C19",
    "CYP2D6", "CYP3A4", "CYP3A5",
}
TRANSPORTERS: Set[str] = {
    "ABCB1",   # P-glycoprotein (P-gp)
    "ABCG2",   # BCRP
    "SLCO1B1", # OATP1B1
    "SLCO1B3", # OATP1B3
    "SLC22A2", # OCT2
}
METABOLIC_NODES: Set[str] = CYP_ENZYMES | TRANSPORTERS

# Human-readable aliases used in prompts
NODE_ALIASES: Dict[str, str] = {
    "ABCB1": "P-gp",
    "ABCG2": "BCRP",
    "SLCO1B1": "OATP1B1",
    "SLCO1B3": "OATP1B3",
    "SLC22A2": "OCT2",
}

# Relationship types that indicate the drug is a substrate of a CYP
SUBSTRATE_ACTIONS = {"substrate", "metabolized by", "metabolised by"}
# Relationship types that indicate the drug inhibits a CYP
INHIBITOR_ACTIONS = {"inhibitor", "inhibits", "strong inhibitor", "moderate inhibitor", "weak inhibitor"}
# Relationship types that indicate the drug induces a CYP
INDUCER_ACTIONS = {"inducer", "induces"}


class DrugBankLoader:
    """Loads and indexes dhimmel/drugbank TSV data for PharmaChess.

    Attributes:
        data_dir:      Path to the directory containing the TSV files.
        _drugs:        Dict[drugbank_id → drug_record]
        _name_index:   Dict[lowercase_name → drugbank_id]
        _interactions: Dict[drugbank_id → List[interaction_record]]
        _cyp_profiles: Dict[drugbank_id → {"substrates", "inhibitors", "inducers"}]
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._drugs: Dict[str, Dict] = {}
        self._name_index: Dict[str, str] = {}
        self._interactions: Dict[str, List[Dict]] = {}
        self._cyp_profiles: Dict[str, Dict[str, List[str]]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Parse all relevant TSV files.  Safe to call multiple times."""
        if self._loaded:
            return
        self._load_drugs()
        self._load_interactions()
        self._load_targets()
        self._loaded = True

    def _load_drugs(self) -> None:
        path = self.data_dir / "drugbank.tsv"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                db_id = row.get("drugbank_id", "")
                name = row.get("name", "").lower()
                self._drugs[db_id] = {
                    "id": db_id,
                    "name": row.get("name", ""),
                    "type": row.get("type", ""),
                    "groups": row.get("groups", ""),
                    "description": row.get("description", "")[:400],
                }
                if name:
                    self._name_index[name] = db_id
                # Also index common synonyms if present
                for syn in row.get("synonyms", "").split("|"):
                    syn = syn.strip().lower()
                    if syn:
                        self._name_index.setdefault(syn, db_id)

    def _load_interactions(self) -> None:
        path = self.data_dir / "drug-interactions.tsv"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                db_id = row.get("drugbank_id", "")
                interaction = {
                    "drug_a": row.get("name", ""),
                    "drug_b": row.get("interacting_drug", ""),
                    "drug_b_id": row.get("interacting_drug_id", ""),
                    "description": row.get("description", "")[:300],
                    "severity": self._infer_severity(row.get("description", "")),
                }
                self._interactions.setdefault(db_id, []).append(interaction)

    def _load_targets(self) -> None:
        """Build CYP profiles from drug-targets.tsv.

        Each row: drugbank_id, drug_name, target_gene_symbol, action, …
        We filter to rows where target_gene_symbol is a CYP enzyme or
        transporter and classify the relationship as substrate/inhibitor/inducer.
        """
        path = self.data_dir / "drug-targets.tsv"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                gene = (row.get("gene_name") or row.get("gene_symbol", "")).upper()
                if gene not in METABOLIC_NODES:
                    continue
                db_id = row.get("drugbank_id", "")
                action = (row.get("action") or row.get("known_action", "")).lower()
                profile = self._cyp_profiles.setdefault(
                    db_id, {"substrates": [], "inhibitors": [], "inducers": []}
                )
                label = NODE_ALIASES.get(gene, gene)
                if any(a in action for a in SUBSTRATE_ACTIONS):
                    if label not in profile["substrates"]:
                        profile["substrates"].append(label)
                elif any(a in action for a in INHIBITOR_ACTIONS):
                    if label not in profile["inhibitors"]:
                        profile["inhibitors"].append(label)
                elif any(a in action for a in INDUCER_ACTIONS):
                    if label not in profile["inducers"]:
                        profile["inducers"].append(label)

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def resolve_name(self, drug_name: str) -> Optional[str]:
        """Return DrugBank ID for a drug name (case-insensitive). None if unknown."""
        return self._name_index.get(drug_name.lower())

    def get_drug_info(self, drug_name: str) -> Optional[Dict]:
        """Return the master drug record for a given name."""
        db_id = self.resolve_name(drug_name)
        return self._drugs.get(db_id) if db_id else None

    def get_cyp_profile(self, drug_name: str) -> Dict[str, List[str]]:
        """Return CYP / transporter profile for a drug.

        Returns:
            {"substrates": [...], "inhibitors": [...], "inducers": [...]}
            Empty lists if the drug is unknown or has no CYP data.
        """
        db_id = self.resolve_name(drug_name)
        if db_id and db_id in self._cyp_profiles:
            return self._cyp_profiles[db_id]
        # Graceful fallback with built-in reference data for common drugs
        return _BUILTIN_CYP_PROFILES.get(drug_name.lower(),
                                          {"substrates": [], "inhibitors": [], "inducers": []})

    def get_interactions(self, drug_name: str,
                          other_drugs: Optional[List[str]] = None) -> List[Dict]:
        """Return known DDIs for a drug, optionally filtered to a co-drug list.

        Each record:
            {"drug_a": str, "drug_b": str, "description": str, "severity": str}
        """
        db_id = self.resolve_name(drug_name)
        all_interactions = self._interactions.get(db_id, [])
        if other_drugs is None:
            return all_interactions
        other_lower = {d.lower() for d in other_drugs}
        return [
            i for i in all_interactions
            if i.get("drug_b", "").lower() in other_lower
        ]

    def build_cyp_occupancy(self, drugs: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """Build the CYP occupancy matrix for a full drug regimen.

        Returns a dict mapping each metabolic node to the drugs competing for it:
            {
              "CYP3A4": {
                "substrates":  ["rivaroxaban", "atorvastatin"],
                "inhibitors":  ["clarithromycin"],
                "inducers":    [],
              },
              ...
            }
        This is the 'territory map' that the LLM agents reason over.
        """
        occupancy: Dict[str, Dict[str, List[str]]] = {
            node: {"substrates": [], "inhibitors": [], "inducers": []}
            for node in METABOLIC_NODES | set(NODE_ALIASES.values())
        }
        for drug in drugs:
            profile = self.get_cyp_profile(drug)
            for node in profile.get("substrates", []):
                occupancy.setdefault(node, {"substrates": [], "inhibitors": [], "inducers": []})
                occupancy[node]["substrates"].append(drug)
            for node in profile.get("inhibitors", []):
                occupancy.setdefault(node, {"substrates": [], "inhibitors": [], "inducers": []})
                occupancy[node]["inhibitors"].append(drug)
            for node in profile.get("inducers", []):
                occupancy.setdefault(node, {"substrates": [], "inhibitors": [], "inducers": []})
                occupancy[node]["inducers"].append(drug)
        # Remove empty nodes to keep prompts concise
        return {k: v for k, v in occupancy.items()
                if any(v[role] for role in ("substrates", "inhibitors", "inducers"))}

    def get_all_interaction_pairs(self, drugs: List[str]) -> List[Dict]:
        """Return all known DDIs between drugs in the regimen (pairwise)."""
        pairs = []
        drug_set = {d.lower() for d in drugs}
        for drug in drugs:
            for interaction in self.get_interactions(drug):
                if interaction.get("drug_b", "").lower() in drug_set:
                    pairs.append(interaction)
        return pairs

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_severity(description: str) -> str:
        desc_lower = description.lower()
        if any(w in desc_lower for w in ("contraindicated", "avoid", "fatal", "life-threatening")):
            return "contraindicated"
        if any(w in desc_lower for w in ("major", "serious", "significantly")):
            return "major"
        if any(w in desc_lower for w in ("moderate", "caution", "monitor")):
            return "moderate"
        return "minor"


# ---------------------------------------------------------------------------
# Built-in CYP profiles for commonly studied drugs
# Used as fallback when dhimmel data files are not available.
# Source: FDA Drug Development and Drug Interactions tables (public domain).
# ---------------------------------------------------------------------------
_BUILTIN_CYP_PROFILES: Dict[str, Dict[str, List[str]]] = {
    # Anticoagulants
    "rivaroxaban":   {"substrates": ["CYP3A4", "P-gp"], "inhibitors": [], "inducers": []},
    "apixaban":      {"substrates": ["CYP3A4", "P-gp"], "inhibitors": [], "inducers": []},
    "warfarin":      {"substrates": ["CYP2C9", "CYP3A4"], "inhibitors": [], "inducers": []},
    "dabigatran":    {"substrates": ["P-gp"], "inhibitors": [], "inducers": []},
    # Antiplatelets
    "aspirin":       {"substrates": [], "inhibitors": [], "inducers": []},
    "clopidogrel":   {"substrates": ["CYP2C19"], "inhibitors": ["CYP2C19"], "inducers": []},
    # Statins
    "atorvastatin":  {"substrates": ["CYP3A4", "OATP1B1"], "inhibitors": [], "inducers": []},
    "simvastatin":   {"substrates": ["CYP3A4", "OATP1B1"], "inhibitors": [], "inducers": []},
    "rosuvastatin":  {"substrates": ["OATP1B1", "BCRP"], "inhibitors": [], "inducers": []},
    # Antibiotics (strong CYP inhibitors / inducers)
    "clarithromycin":{"substrates": ["CYP3A4"], "inhibitors": ["CYP3A4", "P-gp"], "inducers": []},
    "rifampicin":    {"substrates": [], "inhibitors": [], "inducers": ["CYP3A4", "CYP2C9", "P-gp"]},
    "fluconazole":   {"substrates": ["CYP3A4"], "inhibitors": ["CYP2C9", "CYP3A4", "CYP2C19"], "inducers": []},
    # Antiepileptics
    "phenytoin":     {"substrates": ["CYP2C9"], "inhibitors": [], "inducers": ["CYP3A4", "CYP2C9"]},
    "carbamazepine": {"substrates": ["CYP3A4"], "inhibitors": [], "inducers": ["CYP3A4", "CYP2C9"]},
    # Antidepressants
    "fluoxetine":    {"substrates": ["CYP2D6", "CYP2C9"], "inhibitors": ["CYP2D6", "CYP2C19"], "inducers": []},
    "paroxetine":    {"substrates": ["CYP2D6"], "inhibitors": ["CYP2D6"], "inducers": []},
    # Proton pump inhibitors
    "omeprazole":    {"substrates": ["CYP2C19", "CYP3A4"], "inhibitors": ["CYP2C19"], "inducers": []},
    # Antihypertensives
    "amlodipine":    {"substrates": ["CYP3A4"], "inhibitors": [], "inducers": []},
    "metoprolol":    {"substrates": ["CYP2D6"], "inhibitors": [], "inducers": []},
    # Antidiabetics
    "metformin":     {"substrates": ["OCT2"], "inhibitors": [], "inducers": []},
    "glibenclamide": {"substrates": ["CYP2C9", "OATP1B1"], "inhibitors": [], "inducers": []},
}
