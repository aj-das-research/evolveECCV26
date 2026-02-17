"""CYP Metabolic Graph — the PharmaChess 'Game Board'

In Diplomacy, the board is a map of 82 provinces connected by adjacency edges.
In PharmaChess, the board is the human drug metabolism network:

  Nodes (provinces / supply centres):
    - CYP enzymes : CYP1A2, CYP2B6, CYP2C8, CYP2C9, CYP2C19, CYP2D6,
                    CYP3A4, CYP3A5
    - Transporters: P-gp (ABCB1), BCRP (ABCG2), OATP1B1, OATP1B3, OCT2

  Edges (adjacency / interaction):
    - Substrate edge   : drug metabolised via this enzyme/transporter
    - Inhibitor edge   : drug blocks this node (captures territory)
    - Inducer  edge    : drug upregulates this node (affects all substrates)
    - Enzyme-enzyme edges represent shared substrate overlap (co-regulation
      between CYP2C9 and CYP2C19, for instance) — analogous to sea lanes.

  Key supply centres (strategically critical nodes — most drug interactions):
    CYP3A4, CYP2D6, CYP2C9, P-gp

The adjacency matrix produced here is compatible with the existing
graph_convolution.py layer (models/layers/graph_convolution.py) — just
substitute this matrix for the Diplomacy board adjacency matrix to apply
GCN-based reasoning over the metabolic network.
"""
from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Node catalogue
# ---------------------------------------------------------------------------

class NodeType(enum.Enum):
    CYP_ENZYME   = "cyp_enzyme"
    TRANSPORTER  = "transporter"


@dataclass(frozen=True)
class MetabolicNode:
    """A single node in the metabolic territory map."""
    symbol: str          # e.g. "CYP3A4", "ABCB1"
    alias:  str          # Human-readable short name, e.g. "CYP3A4", "P-gp"
    node_type: NodeType
    is_supply_centre: bool = False   # True for the most clinically critical nodes

    def __str__(self) -> str:
        return self.alias


class InteractionType(enum.Enum):
    SUBSTRATE = "substrate"   # Drug is metabolised by this node
    INHIBITOR = "inhibitor"   # Drug inhibits / blocks this node
    INDUCER   = "inducer"     # Drug upregulates this node
    INDUCER_SUBSTRATE = "inducer_substrate"  # Both roles


# ---------------------------------------------------------------------------
# Node registry
# ---------------------------------------------------------------------------

NODES: List[MetabolicNode] = [
    # CYP Enzymes
    MetabolicNode("CYP1A2",  "CYP1A2",  NodeType.CYP_ENZYME),
    MetabolicNode("CYP2B6",  "CYP2B6",  NodeType.CYP_ENZYME),
    MetabolicNode("CYP2C8",  "CYP2C8",  NodeType.CYP_ENZYME),
    MetabolicNode("CYP2C9",  "CYP2C9",  NodeType.CYP_ENZYME,  is_supply_centre=True),
    MetabolicNode("CYP2C19", "CYP2C19", NodeType.CYP_ENZYME),
    MetabolicNode("CYP2D6",  "CYP2D6",  NodeType.CYP_ENZYME,  is_supply_centre=True),
    MetabolicNode("CYP3A4",  "CYP3A4",  NodeType.CYP_ENZYME,  is_supply_centre=True),
    MetabolicNode("CYP3A5",  "CYP3A5",  NodeType.CYP_ENZYME),
    # Transporters
    MetabolicNode("ABCB1",   "P-gp",    NodeType.TRANSPORTER, is_supply_centre=True),
    MetabolicNode("ABCG2",   "BCRP",    NodeType.TRANSPORTER),
    MetabolicNode("SLCO1B1", "OATP1B1", NodeType.TRANSPORTER),
    MetabolicNode("SLCO1B3", "OATP1B3", NodeType.TRANSPORTER),
    MetabolicNode("SLC22A2", "OCT2",    NodeType.TRANSPORTER),
]

NODE_BY_SYMBOL: Dict[str, MetabolicNode] = {n.symbol: n for n in NODES}
NODE_BY_ALIAS:  Dict[str, MetabolicNode] = {n.alias: n for n in NODES}
NODE_INDEX:     Dict[str, int]           = {n.symbol: i for i, n in enumerate(NODES)}
N_NODES = len(NODES)

# Metabolic adjacency: enzymes that share substrate overlap
# (analogous to Diplomacy's sea-lane connections)
_ENZYME_ADJACENCY: List[Tuple[str, str]] = [
    # CYP2C subfamily are co-regulated and share many substrates
    ("CYP2C9",  "CYP2C19"),
    ("CYP2C8",  "CYP2C9"),
    ("CYP2C8",  "CYP2C19"),
    # CYP3A subfamily overlap
    ("CYP3A4",  "CYP3A5"),
    # P-gp and BCRP are co-expressed at blood-brain barrier / gut
    ("ABCB1",   "ABCG2"),
    # OATP transporters work in concert in hepatic uptake
    ("SLCO1B1", "SLCO1B3"),
    # CYP3A4 and P-gp often co-inhibited (e.g. by clarithromycin, ketoconazole)
    ("CYP3A4",  "ABCB1"),
]


# ---------------------------------------------------------------------------
# Graph class
# ---------------------------------------------------------------------------

@dataclass
class CYPGraph:
    """The metabolic territory map for PharmaChess.

    Attributes:
        adjacency_matrix: (N_NODES × N_NODES) binary ndarray — 1 if two
            nodes share metabolic co-regulation or are co-affected by common
            inhibitors.  Drop-in replacement for the Diplomacy board adjacency
            matrix used in graph_convolution.py.
        drug_node_edges: Dict mapping drug_name → {node_alias → InteractionType}
        conflict_edges:  Set of (drug_a, drug_b, node_alias) tuples where
            two drugs compete for the same metabolic node.
    """
    adjacency_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((N_NODES, N_NODES), dtype=np.float32)
    )
    drug_node_edges: Dict[str, Dict[str, InteractionType]] = field(
        default_factory=dict
    )
    conflict_edges: Set[Tuple[str, str, str]] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, cyp_occupancy: Dict[str, Dict[str, List[str]]]) -> "CYPGraph":
        """Build a CYPGraph from a DrugBankLoader occupancy matrix.

        Args:
            cyp_occupancy: Output of DrugBankLoader.build_cyp_occupancy().
                Format: {node_alias: {"substrates": [...], "inhibitors": [...],
                                      "inducers": [...]}}

        Returns:
            CYPGraph with populated adjacency_matrix and conflict_edges.
        """
        graph = cls()
        graph._build_base_adjacency()
        graph._add_drug_edges(cyp_occupancy)
        graph._detect_conflicts(cyp_occupancy)
        return graph

    def _build_base_adjacency(self) -> None:
        """Populate the static enzyme/transporter co-regulation edges."""
        adj = self.adjacency_matrix
        # Self-loops (a node is always adjacent to itself)
        for i in range(N_NODES):
            adj[i, i] = 1.0

        for sym_a, sym_b in _ENZYME_ADJACENCY:
            i = NODE_INDEX.get(sym_a)
            j = NODE_INDEX.get(sym_b)
            if i is not None and j is not None:
                adj[i, j] = 1.0
                adj[j, i] = 1.0  # undirected

    def _add_drug_edges(self,
                         cyp_occupancy: Dict[str, Dict[str, List[str]]]) -> None:
        """Record which drugs occupy which nodes and how."""
        for node_alias, roles in cyp_occupancy.items():
            node = NODE_BY_ALIAS.get(node_alias)
            if node is None:
                continue
            for drug in roles.get("substrates", []):
                self.drug_node_edges.setdefault(drug, {})[node_alias] = InteractionType.SUBSTRATE
            for drug in roles.get("inhibitors", []):
                existing = self.drug_node_edges.setdefault(drug, {}).get(node_alias)
                if existing == InteractionType.INDUCER:
                    self.drug_node_edges[drug][node_alias] = InteractionType.INDUCER_SUBSTRATE
                else:
                    self.drug_node_edges[drug][node_alias] = InteractionType.INHIBITOR
            for drug in roles.get("inducers", []):
                self.drug_node_edges.setdefault(drug, {})[node_alias] = InteractionType.INDUCER

    def _detect_conflicts(self,
                           cyp_occupancy: Dict[str, Dict[str, List[str]]]) -> None:
        """Detect territorial conflicts: inhibitor vs substrate on same node.

        A conflict occurs when:
          - Drug A inhibits node X  AND
          - Drug B is a substrate of node X

        This means Drug A's inhibition affects Drug B's metabolism — the
        classic "territory capture" in the Diplomacy analogy.
        """
        for node_alias, roles in cyp_occupancy.items():
            inhibitors = set(roles.get("inhibitors", []))
            substrates = set(roles.get("substrates", []))
            for inhib in inhibitors:
                for substr in substrates:
                    if inhib != substr:
                        # Canonical ordering to avoid duplicates
                        a, b = (inhib, substr) if inhib < substr else (substr, inhib)
                        self.conflict_edges.add((a, b, node_alias))

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_conflicts_for_drug(self, drug_name: str) -> List[Dict]:
        """Return all territorial conflicts involving a specific drug.

        Returns:
            List of {"drug_a": ..., "drug_b": ..., "node": ...,
                     "type": "inhibitor_blocks_substrate" | "substrate_blocked"}
        """
        results = []
        drug_lower = drug_name.lower()
        for a, b, node in self.conflict_edges:
            if a.lower() == drug_lower or b.lower() == drug_lower:
                # Determine which role this drug plays
                drug_role = self.drug_node_edges.get(drug_name, {}).get(node)
                conflict_type = (
                    "inhibitor_blocks_substrate"
                    if drug_role == InteractionType.INHIBITOR
                    else "substrate_blocked"
                )
                results.append({
                    "drug_a": a, "drug_b": b, "node": node,
                    "conflict_type": conflict_type,
                })
        return results

    def get_conflict_score(self, regimen: List[str]) -> float:
        """Return a normalised conflict score for the entire regimen.

        0.0 = no conflicts, 1.0 = maximum possible conflicts.
        Analogous to the supply-centre count in Diplomacy.
        """
        regimen_set = {d.lower() for d in regimen}
        regimen_conflicts = [
            c for c in self.conflict_edges
            if c[0].lower() in regimen_set and c[1].lower() in regimen_set
        ]
        # Number of possible pairwise conflicts: n*(n-1)/2 × N_NODES
        max_conflicts = max(len(regimen) * (len(regimen) - 1) // 2 * N_NODES, 1)
        return min(len(regimen_conflicts) / max_conflicts, 1.0)

    def supply_centre_pressure(self, regimen: List[str]) -> Dict[str, int]:
        """Return how many drugs are competing for each supply-centre node.

        Higher pressure = higher clinical risk.  Analogous to multiple armies
        fighting over a single supply centre in Diplomacy.
        """
        supply_centres = [n.alias for n in NODES if n.is_supply_centre]
        pressure = {sc: 0 for sc in supply_centres}
        for drug in regimen:
            for node_alias, itype in self.drug_node_edges.get(drug, {}).items():
                if node_alias in pressure:
                    pressure[node_alias] += 1
        return pressure

    def to_adjacency_dict(self) -> Dict[str, List[str]]:
        """Return the adjacency structure in the same format as Diplomacy's
        province adjacency dict (used in prompt.txt) — for LLM context.
        """
        result: Dict[str, List[str]] = {}
        for i, node in enumerate(NODES):
            neighbours = [
                NODES[j].alias
                for j in range(N_NODES)
                if self.adjacency_matrix[i, j] > 0 and j != i
            ]
            result[node.alias] = neighbours
        return result

    def summary(self) -> str:
        """Return a human-readable summary for use in prompts."""
        lines = [
            f"Metabolic territory map — {N_NODES} nodes, "
            f"{len(self.conflict_edges)} active conflicts"
        ]
        pressure = {}
        for a, b, node in self.conflict_edges:
            pressure[node] = pressure.get(node, 0) + 1
        for node, count in sorted(pressure.items(), key=lambda x: -x[1])[:5]:
            is_sc = NODE_BY_ALIAS.get(node, MetabolicNode("", "", NodeType.CYP_ENZYME)).is_supply_centre
            sc_marker = " [KEY NODE]" if is_sc else ""
            lines.append(f"  {node}{sc_marker}: {count} drug(s) in conflict")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialise the graph for caching or logging."""
        return json.dumps({
            "n_nodes": N_NODES,
            "nodes": [{"symbol": n.symbol, "alias": n.alias,
                       "type": n.node_type.value,
                       "supply_centre": n.is_supply_centre} for n in NODES],
            "conflict_edges": list(self.conflict_edges),
            "drug_node_edges": {
                drug: {node: itype.value for node, itype in edges.items()}
                for drug, edges in self.drug_node_edges.items()
            },
        }, indent=2)


# ---------------------------------------------------------------------------
# Convenience: build a graph directly from a drug list + DrugBankLoader
# ---------------------------------------------------------------------------

def build_graph_for_regimen(drugs: List[str], loader) -> CYPGraph:
    """Shortcut: build a CYPGraph from a drug list using a DrugBankLoader.

    Args:
        drugs:  List of generic drug names.
        loader: DrugBankLoader instance (must have .load() called).

    Returns:
        Populated CYPGraph ready for conflict analysis.
    """
    occupancy = loader.build_cyp_occupancy(drugs)
    return CYPGraph.build(occupancy)
