# PharmaChess — Setup and Usage Guide

**Polypharmacy Risk Assessment via Self-Evolving Multi-Agent LLM Reasoning**

Adapted from the Richelieu framework (NeurIPS 2024 — *Self-Evolving LLM-Based Agents for AI Diplomacy*).
Paper: https://arxiv.org/abs/2407.06813

---

## Table of Contents

1. [Concept Overview](#1-concept-overview)
2. [Prerequisites](#2-prerequisites)
3. [API Setup](#3-api-setup)
   - 3.1 [OpenAI API (required)](#31-openai-api-required)
   - 3.2 [OpenFDA FAERS API (strongly recommended)](#32-openfda-faers-api-strongly-recommended)
   - 3.3 [PubMed E-utilities / NCBI API (recommended)](#33-pubmed-e-utilities--ncbi-api-recommended)
   - 3.4 [DrugBank Data — dhimmel TSV files (recommended)](#34-drugbank-data--dhimmel-tsv-files-recommended)
4. [Environment Setup](#4-environment-setup)
5. [Configuration Reference](#5-configuration-reference)
6. [Running PharmaChess](#6-running-pharmachess)
   - 6.1 [Single regimen evaluation](#61-single-regimen-evaluation)
   - 6.2 [Custom regimen](#62-custom-regimen)
   - 6.3 [Self-play evolution loop](#63-self-play-evolution-loop)
   - 6.4 [Using OpenAI-compatible APIs](#64-using-openai-compatible-apis)
7. [Understanding the Output](#7-understanding-the-output)
8. [Module Architecture](#8-module-architecture)
9. [Richelieu ↔ PharmaChess Analogy Reference](#9-richelieu--pharmachess-analogy-reference)
10. [Extending PharmaChess](#10-extending-pharmachess)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Concept Overview

PharmaChess treats polypharmacy risk assessment as a **multi-agent territorial control problem**.

The human body's drug metabolism network (CYP enzymes, drug transporters) is the game board.
Each drug in a patient's regimen is an autonomous **agent** (analogous to a Diplomacy power).
When Drug A inhibits CYP3A4 and Drug B is metabolised by CYP3A4, Drug A has **captured that metabolic territory** and Drug B's plasma concentration rises — a direct clinical risk.

A **coordinator agent** (Richelieu's master reasoner) synthesises intelligence from:
- **FAERS** — real-world adverse event co-report statistics
- **DrugBank** — known drug-drug interaction severity and CYP mechanism
- **PubMed** — published case reports and mechanistic literature
- **Memory bank** — past evaluated regimens retrieved by CYP-pathway similarity

The system **self-evolves** by running across diverse drug combinations, storing outcomes in the memory bank, and using them to improve future reasoning — no human-labelled training data required.

---

## 2. Prerequisites

- Python **3.8** (matches the conda environment in the main repo)
- Dependencies already present in `requirements.txt`:
  - `langchain==0.2.17`, `langchain-openai==0.1.25`, `openai==1.69.0`
  - `aiohttp`, `numpy`, `networkx`
- Internet access for FAERS and PubMed API calls

---

## 3. API Setup

### 3.1 OpenAI API (required)

The drug agents and coordinator use any OpenAI-compatible chat completion endpoint.

**Get an API key:**
1. Go to https://platform.openai.com/api-keys
2. Click **Create new secret key**
3. Copy the key (starts with `sk-`)

**Set the environment variable:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Recommended model:**  `gpt-4o` (default).
Any model that supports JSON-mode chat completions works, including `gpt-4-turbo`, `gpt-4o-mini`, `o1-mini`.

```bash
export OPENAI_MODEL="gpt-4o"        # default if not set
```

---

### 3.2 OpenFDA FAERS API (strongly recommended)

The FAERS client hits `https://api.fda.gov/drug/event.json`.

**Without a key**: 240 requests/minute — adequate for single evaluations.
**With a key**: 120,000 requests/day — required for self-play loops.

**Get a free API key:**
1. Go to https://open.fda.gov/apis/authentication/
2. Enter your email and agree to terms
3. Check your email for the key (arrives immediately)

```bash
export OPENFDA_API_KEY="your_openfda_key"
```

**Verify it works (paste in browser or curl):**
```bash
curl "https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:\"rivaroxaban\"+AND+patient.drug.openfda.generic_name:\"aspirin\"&limit=3&api_key=$OPENFDA_API_KEY"
```

---

### 3.3 PubMed E-utilities / NCBI API (recommended)

Used by `PubMedClient` for literature evidence.

**Without a key**: 3 requests/second.
**With a key**: 10 requests/second.

**Get a free API key:**
1. Create an NCBI account at https://www.ncbi.nlm.nih.gov/account/
2. After login, go to **Account Settings → API Key Management**
3. Click **Create an API Key**

```bash
export NCBI_API_KEY="your_ncbi_key"
export NCBI_EMAIL="your@email.com"   # good practice per NCBI guidelines
```

---

### 3.4 DrugBank Data — dhimmel TSV files (recommended)

Pre-parsed DrugBank data from https://github.com/dhimmel/drugbank (CC BY-NC 4.0).
Free to download — no DrugBank account required.

**Download:**
```bash
git clone https://github.com/dhimmel/drugbank.git /path/to/drugbank-data
```

The loader reads these files from the `data/` subdirectory:

| File | Contents |
|---|---|
| `data/drugbank.tsv` | Master drug table (ID, name, type, description) |
| `data/drug-interactions.tsv` | Pairwise DDI records with descriptions |
| `data/drug-targets.tsv` | Drug → gene target mappings (used to build CYP profiles) |
| `data/drug-categories.tsv` | ATC pharmacological category hierarchy |

```bash
export DRUGBANK_DATA_DIR="/path/to/drugbank-data/data"
```

**Fallback without download:**
The `DrugBankLoader` ships with built-in CYP profiles for 20 commonly studied drugs
(rivaroxaban, warfarin, atorvastatin, clarithromycin, metformin, fluoxetine, etc.).
The system runs correctly without the TSV files — it simply uses the built-in profiles
instead. Set `DRUGBANK_DATA_DIR` to any empty directory or omit it entirely.

---

## 4. Environment Setup

Use the existing repo conda environment:

```bash
# From the project root (evolveECCV26/)
conda activate diplomacy

# All required packages are already in requirements.txt
# If starting fresh:
pip install --no-deps -r requirements.txt
```

**Verify the installation:**
```bash
python -c "from diplomacy_research.pharmachess.agents.base_drug_agent import OpenAIConfig; print('OK')"
```

---

## 5. Configuration Reference

All configuration is via environment variables. None require code changes.

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI or compatible API key |
| `OPENAI_BASE_URL` | No | OpenAI default | Override for compatible APIs (Together AI, Groq, vLLM, etc.) |
| `OPENAI_MODEL` | No | `gpt-4o` | Model name for all agents and coordinator |
| `OPENFDA_API_KEY` | No | — | 120K req/day quota (vs 240/min without) |
| `NCBI_API_KEY` | No | — | 10 req/s PubMed rate (vs 3/s without) |
| `NCBI_EMAIL` | No | `pharmachess@example.com` | Email for NCBI courtesy identification |
| `DRUGBANK_DATA_DIR` | No | — | Path to dhimmel/drugbank `data/` directory |
| `PHARMACHESS_MEMORY_PATH` | No | `pharmachess_memory.jsonl` | Path for persistent memory bank |

---

## 6. Running PharmaChess

All commands are run from the project root (`evolveECCV26/`).

### 6.1 Single regimen evaluation

Evaluates the **default high-risk regimen**: rivaroxaban + aspirin + atorvastatin + clarithromycin.
(clarithromycin is a strong CYP3A4 + P-gp inhibitor that dramatically increases rivaroxaban and atorvastatin exposure)

```bash
conda activate diplomacy

export OPENAI_API_KEY="sk-..."
export OPENFDA_API_KEY="your_openfda_key"   # optional but recommended
export NCBI_API_KEY="your_ncbi_key"         # optional

python diplomacy_research/pharmachess/tests/run_pharmachess.py
```

**What happens:**
1. DrugBank CYP profiles are loaded (TSVs or built-in fallback)
2. One `OpenAIDrugAgent` is created for each of the 4 drugs
3. `RegimenCoordinator` is initialised
4. The environment queries FAERS for the drug combination
5. Each drug agent assesses its own metabolic risk (parallel LLM calls)
6. The coordinator synthesises a final risk verdict
7. Agents propose actions (HOLD / DECREASE_DOSE / FLAG_INTERACTION / etc.)
8. The environment applies actions and recomputes state
9. Episode repeats until converged or 5 turns reached
10. The episode is stored in the memory bank (`pharmachess_memory.jsonl`)

---

### 6.2 Custom regimen

Pass any generic drug names with `--drugs`:

```bash
# Classic warfarin + fluconazole CYP2C9 interaction
python diplomacy_research/pharmachess/tests/run_pharmachess.py \
    --drugs warfarin fluconazole omeprazole

# Clopidogrel + PPI CYP2C19 interaction
python diplomacy_research/pharmachess/tests/run_pharmachess.py \
    --drugs clopidogrel omeprazole aspirin

# High-risk cardiac polypharmacy
python diplomacy_research/pharmachess/tests/run_pharmachess.py \
    --drugs warfarin aspirin omeprazole amiodarone

# Safer elderly regimen
python diplomacy_research/pharmachess/tests/run_pharmachess.py \
    --drugs apixaban aspirin metformin amlodipine
```

**Supported drugs (built-in CYP profiles, no DrugBank TSVs needed):**

| Drug | Key CYP roles |
|---|---|
| rivaroxaban | CYP3A4 substrate, P-gp substrate |
| apixaban | CYP3A4 substrate, P-gp substrate |
| warfarin | CYP2C9 substrate, CYP3A4 substrate |
| dabigatran | P-gp substrate |
| aspirin | No major CYP pathway |
| clopidogrel | CYP2C19 substrate + inhibitor |
| atorvastatin | CYP3A4 substrate, OATP1B1 substrate |
| simvastatin | CYP3A4 substrate, OATP1B1 substrate |
| rosuvastatin | OATP1B1 substrate, BCRP substrate |
| clarithromycin | CYP3A4 substrate + **strong inhibitor**, P-gp inhibitor |
| rifampicin | CYP3A4 inducer, CYP2C9 inducer, P-gp inducer |
| fluconazole | CYP2C9 inhibitor, CYP3A4 inhibitor, CYP2C19 inhibitor |
| fluoxetine | CYP2D6 substrate + inhibitor, CYP2C19 inhibitor |
| paroxetine | CYP2D6 substrate + inhibitor |
| metoprolol | CYP2D6 substrate |
| omeprazole | CYP2C19 substrate + inhibitor |
| phenytoin | CYP2C9 substrate + inducer, CYP3A4 inducer |
| carbamazepine | CYP3A4 substrate + inducer |
| amlodipine | CYP3A4 substrate |
| metformin | OCT2 substrate |
| glibenclamide | CYP2C9 substrate, OATP1B1 substrate |

Any drug not in this list can still be evaluated — the CYP profile will be empty and the
agent will rely on FAERS and DrugBank TSV data (if available) rather than the built-in table.

---

### 6.3 Self-play evolution loop

Runs multiple episodes across the built-in regimen pool to grow the memory bank.
This is the **Richelieu self-evolution mechanism** — no human annotation required.

```bash
# 12 episodes (covers all built-in clinical scenarios once)
python diplomacy_research/pharmachess/tests/run_pharmachess.py --self-play 12

# 24 episodes (each scenario twice, memory grows richer)
python diplomacy_research/pharmachess/tests/run_pharmachess.py --self-play 24

# Persist to a specific memory path across runs
python diplomacy_research/pharmachess/tests/run_pharmachess.py \
    --self-play 24 \
    --memory-path /path/to/my_pharmachess_memory.jsonl
```

**Built-in self-play regimen pool (12 scenarios):**

| Regimen | Clinical scenario |
|---|---|
| warfarin + fluconazole + omeprazole | CYP2C9 conflict → warfarin toxicity |
| clopidogrel + omeprazole + aspirin | CYP2C19 conflict → antiplatelet failure |
| metoprolol + fluoxetine + paroxetine | CYP2D6 conflict → beta-blocker overdose |
| simvastatin + clarithromycin + amlodipine | CYP3A4 conflict → statin myopathy |
| rivaroxaban + rifampicin | P-gp + CYP3A4 induction → anticoagulation failure |
| apixaban + aspirin + metformin | Low-interaction reference |
| warfarin + aspirin + omeprazole + amiodarone | Classic high-risk polypharmacy |
| phenytoin + warfarin + fluconazole | CYP2C9 induction + inhibition conflict |
| carbamazepine + atorvastatin + metoprolol | CYP3A4/2D6 induction |
| dabigatran + clarithromycin + aspirin | P-gp inhibition → bleeding risk |
| metformin + amlodipine + rosuvastatin | Low-interaction reference |
| rivaroxaban + aspirin + atorvastatin + clarithromycin | Default scenario |

After a self-play run, memory statistics are printed:
```
Memory stats: {
  "size": 12,
  "avg_risk_score": 0.34,
  "max_risk_score": 0.81,
  "outcome_breakdown": {"high_risk": 4, "moderate": 5, "safe": 3},
  "sources": {"self_play": 12, ...}
}
```

Subsequent runs on the same regimens will retrieve memory precedents and improve
the depth of reasoning — the system gets smarter with each iteration.

---

### 6.4 Using OpenAI-compatible APIs

The `OpenAIConfig.base_url` field passes through to LangChain's `ChatOpenAI(openai_api_base=...)`,
making PharmaChess compatible with any OpenAI-protocol server.

**Together AI:**
```bash
export OPENAI_API_KEY="your_together_key"
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export OPENAI_MODEL="meta-llama/Llama-3-70b-chat-hf"
python diplomacy_research/pharmachess/tests/run_pharmachess.py --drugs warfarin fluconazole
```

**Groq:**
```bash
export OPENAI_API_KEY="your_groq_key"
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export OPENAI_MODEL="llama3-70b-8192"
python diplomacy_research/pharmachess/tests/run_pharmachess.py
```

**Local vLLM / Ollama:**
```bash
export OPENAI_API_KEY="not-used"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="your-local-model"
python diplomacy_research/pharmachess/tests/run_pharmachess.py
```

**Quiet mode** (suppress turn-by-turn output, only show final result):
```bash
python diplomacy_research/pharmachess/tests/run_pharmachess.py --quiet
```

**Full CLI reference:**
```
usage: run_pharmachess.py [-h]
                          [--drugs DRUG [DRUG ...]]
                          [--self-play N]
                          [--drugbank-dir PATH]
                          [--memory-path PATH]
                          [--model MODEL]
                          [--quiet]

  --drugs       Generic drug names to evaluate (space-separated)
  --self-play N Run N episodes from the built-in self-play pool
  --drugbank-dir  Path to dhimmel/drugbank data/ directory
  --memory-path   Path to persist the memory bank (default: pharmachess_memory.jsonl)
  --model         OpenAI model name (default: gpt-4o)
  --quiet         Suppress verbose per-turn output
```

---

## 7. Understanding the Output

### Per-turn console output

```
════════════════════════════════════════════════════════════
PharmaChess Episode
Regimen: rivaroxaban, aspirin, atorvastatin, clarithromycin
════════════════════════════════════════════════════════════
  Turn 0 | Risk: 0.421 | FAERS: 0.002 | Conflicts: 3 | DDIs: 2

── Turn 1 ──
  [rivaroxaban]    Risk: HIGH (0.72) | Conflicts: ['CYP3A4', 'P-gp']
  [aspirin]        Risk: MODERATE (0.35) | Conflicts: []
  [atorvastatin]   Risk: HIGH (0.68) | Conflicts: ['CYP3A4']
  [clarithromycin] Risk: HIGH (0.81) | Conflicts: ['CYP3A4', 'P-gp']

  COORDINATOR VERDICT: HIGH (score=0.654)
  Primary concern: clarithromycin inhibits CYP3A4 → rivaroxaban and atorvastatin accumulation risk

  [clarithromycin] → FLAG_INTERACTION
  [atorvastatin]   → DECREASE_DOSE

  Risk delta: -0.0231 | Turn: 1
```

| Field | Meaning |
|---|---|
| `Risk: 0.421` | Weighted aggregate of FAERS (40%) + DrugBank DDI severity (35%) + CYP graph conflict (25%) |
| `FAERS: 0.002` | Adverse event co-report rate for this combination in FAERS |
| `Conflicts: 3` | Number of inhibitor→substrate territorial conflicts in the CYP graph |
| `DDIs: 2` | Known pairwise drug-drug interactions from DrugBank |
| `Risk delta` | Change in overall risk score this turn (negative = risk increased, positive = improved) |

### Final JSON result

```json
{
  "regimen": ["rivaroxaban", "aspirin", "atorvastatin", "clarithromycin"],
  "final_verdict": {
    "overall_risk_level": "HIGH",
    "overall_risk_score": 0.621,
    "primary_concern": "clarithromycin inhibits CYP3A4 → rivaroxaban accumulation risk",
    "evidence_synthesis": {
      "faers_weight": "MODERATE",
      "mechanism_weight": "STRONG",
      "literature_weight": "WEAK"
    },
    "priority_actions": [
      {"drug": "clarithromycin", "action": "SWITCH", "rationale": "Azithromycin is a weaker CYP3A4 inhibitor and a safer alternative"},
      {"drug": "atorvastatin",   "action": "DECREASE_DOSE", "rationale": "Reduce statin dose by 50% until CYP3A4 inhibitor is removed"}
    ],
    "reflection": "Similar to case #3 in memory bank (simvastatin + clarithromycin). Statin toxicity risk confirmed by 3 precedents.",
    "coordinator_reasoning": "..."
  },
  "episode_turns": 3,
  "memory_entries_added": 1
}
```

### Memory bank file (`pharmachess_memory.jsonl`)

Each line is a JSON object representing one evaluated episode:
```json
{
  "entry_id": "a3f7c2d91e4b8f01",
  "regimen": ["aspirin", "atorvastatin", "clarithromycin", "rivaroxaban"],
  "cyp_pathways": ["CYP3A4", "P-gp"],
  "risk_score": 0.621,
  "faers_adr_rate": 0.002,
  "conflict_score": 0.375,
  "agent_actions": {"clarithromycin": "SWITCH", "atorvastatin": "DECREASE_DOSE"},
  "outcome_label": "high_risk",
  "explanation": "clarithromycin inhibits CYP3A4 → rivaroxaban and atorvastatin accumulation",
  "eval_score": 0.742,
  "source": "self_play"
}
```

Outcome labels: `safe` (risk < 0.25), `moderate` (0.25–0.50), `high_risk` (0.50–0.75), `contraindicated` (> 0.75).

---

## 8. Module Architecture

```
diplomacy_research/pharmachess/
│
├── agents/                          # Drug agent layer
│   ├── base_drug_agent.py           # Abstract BaseDrugAgent + OpenAIConfig
│   └── openai_drug_agent.py         # Concrete OpenAI-compatible agent
│
├── data/                            # External data clients
│   ├── faers_client.py              # OpenFDA FAERS API (async, with PRR/ROR/IC)
│   ├── drugbank_loader.py           # dhimmel/drugbank TSV loader + built-in CYP profiles
│   └── pubmed_client.py             # PubMed E-utilities async client
│
├── env/                             # Game environment
│   └── polypharmacy_env.py          # PolypharmacyEnv + PolypharmacyState
│
├── graph/                           # Metabolic territory map
│   └── cyp_graph.py                 # CYPGraph (13 nodes, adjacency matrix, conflict detection)
│
├── memory/                          # Richelieu-style memory bank
│   └── polypharmacy_memory.py       # PolypharmacyMemory + MemoryEntry + Jaccard retrieval
│
├── prompts/                         # LLM prompt templates
│   └── drug_interaction_prompt.txt  # Coordinator prompt (territory map + evidence synthesis)
│
├── tests/                           # Orchestration scripts
│   └── run_pharmachess.py           # Main entry point + RegimenCoordinator + self-play loop
│
└── DOCUMENTATION.md                 # This file
```

### Data flow through one evaluation turn

```
PolypharmacyEnv.reset()
  │
  ├─ DrugBankLoader.build_cyp_occupancy(drugs)   → CYP territory map
  ├─ DrugBankLoader.get_all_interaction_pairs()  → Known DDI list
  ├─ CYPGraph.build(occupancy)                   → Conflict detection
  ├─ FAERSClient.query_regimen(drugs)            → FAERS ADR rate + top signals
  └─ PolypharmacyMemory.retrieve_similar()       → Historical precedents
           │
           ▼
    PolypharmacyState  ──────────────────────────────────────┐
           │                                                 │
           ▼                                                 │
  [For each drug in parallel]                                │
  OpenAIDrugAgent.assess_interaction_risk(state)             │ coordinator uses
        → LLM risk assessment JSON                           │ full state + all
  OpenAIDrugAgent.propose_action(state, assessment)          │ agent reports
        → action string (HOLD / DECREASE_DOSE / ...)        │
           │                                                 │
           ▼                                                 │
  RegimenCoordinator.synthesise(state) ◄───────────────────-┘
        → Final verdict JSON (risk level, priority actions, reflection)
           │
           ▼
  PolypharmacyEnv.step(agent_actions)
        → new state, reward, info
           │
           ▼
  PolypharmacyMemory.self_play_update([episode_log])
        → memory bank updated, persisted to disk
```

---

## 9. Richelieu ↔ PharmaChess Analogy Reference

| Richelieu / Diplomacy | PharmaChess | Implementation |
|---|---|---|
| 7 powers (Austria, England…) | Drugs in polypharmacy regimen | `OpenAIDrugAgent` per drug |
| Supply centre | CYP enzyme / transporter (CYP3A4, P-gp…) | `MetabolicNode` in `cyp_graph.py` |
| Key supply centres | CYP3A4, CYP2D6, CYP2C9, P-gp | `is_supply_centre=True` flag |
| Territory capture | Inhibitor drug blocks substrate drug's metabolism | `CYPGraph._detect_conflicts()` |
| Province adjacency matrix | Enzyme co-regulation adjacency matrix | `CYPGraph.adjacency_matrix` (compatible with `graph_convolution.py`) |
| Game state proto | `PolypharmacyState` dataclass | `env/polypharmacy_env.py` |
| `game.process()` | `env.step(agent_actions)` | Applies actions, recomputes risk |
| Supply centre reward | −P(ADR\|regimen) | `_compute_overall_risk()` |
| Richelieu master agent | `RegimenCoordinator` | `tests/run_pharmachess.py` |
| Memory bank (Richelieu §3.3) | `PolypharmacyMemory` | `memory/polypharmacy_memory.py` |
| Jaccard state similarity | Jaccard on CYP pathway sets | `retrieve_similar()` |
| Self-play evolution (§3.4) | `self_play_update()` from FAERS temporal data | No human annotation needed |
| High-evaluative-score entries | Extreme risk entries (very safe or very risky) | `eval_score = abs(risk - 0.5) * 2` |
| Diplomatic negotiation | Evidence synthesis / arbitration | Coordinator prompt `drug_interaction_prompt.txt` |
| Social reasoning | Bradford Hill causality weighting | FAERS 40% + DrugBank 35% + CYP graph 25% |

---

## 10. Extending PharmaChess

### Add a new drug to the built-in profiles

Edit `diplomacy_research/pharmachess/data/drugbank_loader.py`, section `_BUILTIN_CYP_PROFILES`:

```python
"ibuprofen": {"substrates": ["CYP2C9"], "inhibitors": ["CYP2C9"], "inducers": []},
"tramadol":  {"substrates": ["CYP2D6", "CYP3A4"], "inhibitors": [], "inducers": []},
```

### Add a new clinical scenario to the self-play pool

Edit `SELF_PLAY_REGIMENS` in `tests/run_pharmachess.py`:

```python
SELF_PLAY_REGIMENS.append(["tramadol", "fluoxetine", "alprazolam"])  # serotonin + CYP2D6
```

### Use the DrugBankLoader in your own code

```python
from diplomacy_research.pharmachess.data.drugbank_loader import DrugBankLoader

loader = DrugBankLoader(data_dir="/path/to/dhimmel/data")
loader.load()

profile = loader.get_cyp_profile("warfarin")
# {"substrates": ["CYP2C9", "CYP3A4"], "inhibitors": [], "inducers": []}

occupancy = loader.build_cyp_occupancy(["warfarin", "fluconazole", "omeprazole"])
# {"CYP2C9": {"substrates": ["warfarin"], "inhibitors": ["fluconazole"], ...}, ...}

ddis = loader.get_interactions("warfarin", other_drugs=["fluconazole"])
# [{"drug_a": "warfarin", "drug_b": "fluconazole", "severity": "contraindicated", ...}]
```

### Query FAERS directly

```python
import asyncio
from diplomacy_research.pharmachess.data.faers_client import FAERSClient

async def example():
    async with FAERSClient(api_key="your_openfda_key") as client:
        result = await client.query_drug_pair("warfarin", "fluconazole")
        print(f"Reports: {result['report_count']}")
        print(f"PRR: {result['prr']}, ROR: {result['ror']}, IC: {result['ic']}")
        print(f"Top reactions: {result['top_reactions'][:3]}")

asyncio.run(example())
```

### Connect the CYP graph to the existing GCN layer

The adjacency matrix from `CYPGraph` is a drop-in replacement for the Diplomacy board adjacency used in `models/layers/graph_convolution.py`:

```python
from diplomacy_research.pharmachess.graph.cyp_graph import CYPGraph
from diplomacy_research.pharmachess.data.drugbank_loader import DrugBankLoader

loader = DrugBankLoader(data_dir=...)
loader.load()

drugs = ["rivaroxaban", "atorvastatin", "clarithromycin"]
occupancy = loader.build_cyp_occupancy(drugs)
graph = CYPGraph.build(occupancy)

# graph.adjacency_matrix is a (13, 13) float32 numpy array
# Pass it where the Diplomacy board adjacency matrix is expected:
adj_matrix = graph.adjacency_matrix   # shape: (N_NODES, N_NODES)
```

---

## 11. Troubleshooting

### `ModuleNotFoundError: No module named 'diplomacy_research'`

Run from the project root (`evolveECCV26/`), or add the root to `PYTHONPATH`:
```bash
export PYTHONPATH="/path/to/evolveECCV26:$PYTHONPATH"
```

### `ERROR: OPENAI_API_KEY environment variable not set`

The script enforces this. Set it before running:
```bash
export OPENAI_API_KEY="sk-..."
```

### FAERS returns empty results / `404`

OpenFDA returns HTTP 404 (not an error) when the exact drug name combination has zero reports.
The client handles this gracefully and returns `report_count: 0`. This is expected for
uncommon drug combinations or uncommon spelling variants.

Try the query manually to debug:
```bash
curl "https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:\"yourdrugname\"&limit=1"
```
If the drug name is not found, try common brand names or verify spelling against the FDA label.

### Rate limit errors from OpenAI (`429`)

The agents add 0.2s delays between calls. For large self-play loops (> 50 episodes)
with many drugs per regimen, reduce concurrent load:
```bash
python run_pharmachess.py --self-play 12   # batch of 12 is safe with gpt-4o
```

### `DrugBank TSVs not found — using built-in CYP profiles as fallback`

This is an informational message, not an error. The system runs correctly on built-in
profiles. Download the TSV files (see §3.4) for more comprehensive drug coverage.

### Memory bank grows too large

The `PolypharmacyMemory` evicts lowest-`eval_score` entries when the cap (`max_size=10000`)
is reached. If you want a smaller cap, instantiate directly:
```python
memory = PolypharmacyMemory(persist_path="my_memory.jsonl", max_size=500)
```

---

*PharmaChess is a research prototype.
It is not a medical device and should not be used for clinical decision-making.*
