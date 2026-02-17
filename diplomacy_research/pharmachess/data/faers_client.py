"""FAERS Client — OpenFDA Drug Adverse Event API

Wraps the OpenFDA FAERS endpoint (https://api.fda.gov/drug/event.json) to:
  1. Fetch adverse event co-reports for arbitrary drug pairs / regimens.
  2. Compute disproportionality statistics (PRR, ROR, IC) — the standard
     pharmacovigilance signal-detection measures used alongside LLM reasoning.
  3. Retrieve the top adverse event terms (MedDRA PTs) for a given drug combo.

Rate limits:
  - Without API key : 240 requests / minute
  - With API key    : 120 000 requests / day  (register free at open.fda.gov)

Set OPENFDA_API_KEY in the environment to unlock the higher quota.
"""
import asyncio
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp


_BASE_URL = "https://api.fda.gov/drug/event.json"
_DEFAULT_LIMIT = 10
# Total reports in FAERS (approximate — used as N for disproportionality)
_FAERS_TOTAL_REPORTS = 20_000_000


class FAERSClient:
    """Async client for the OpenFDA drug adverse event API.

    Example:
        async with FAERSClient(api_key=os.environ["OPENFDA_API_KEY"]) as client:
            result = await client.query_drug_pair("rivaroxaban", "aspirin")
            print(result["report_count"], result["prr"])
    """

    def __init__(self, api_key: Optional[str] = None,
                 requests_per_second: float = 3.0):
        self.api_key = api_key or os.environ.get("OPENFDA_API_KEY")
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    async def query_drug_pair(self, drug_a: str, drug_b: str,
                               limit: int = _DEFAULT_LIMIT) -> Dict[str, Any]:
        """Return adverse event stats for a specific drug-drug co-exposure.

        Uses the exact query pattern from the OpenFDA documentation:
          patient.drug.openfda.generic_name:"drug_a" AND
          patient.drug.openfda.generic_name:"drug_b"

        Returns:
            {
              "drug_a": str,
              "drug_b": str,
              "report_count": int,
              "top_reactions": List[{"term": str, "count": int}],
              "prr": float,           # Proportional Reporting Ratio
              "ror": float,           # Reporting Odds Ratio
              "ic": float,            # Information Component
              "serious_count": int,   # reports with serious outcomes
              "fatal_count":  int,    # reports with fatal outcome
            }
        """
        search = (
            f'patient.drug.openfda.generic_name:"{drug_a.lower()}"'
            f'+AND+patient.drug.openfda.generic_name:"{drug_b.lower()}"'
        )
        combo_data = await self._fetch(search=search, limit=limit)
        combo_count = combo_data.get("meta", {}).get("results", {}).get("total", 0)

        # Individual counts needed for PRR / ROR
        drug_a_count = await self._get_drug_count(drug_a)
        drug_b_count = await self._get_drug_count(drug_b)

        top_reactions = self._extract_top_reactions(combo_data)
        serious, fatal = await self._count_serious_fatal(drug_a, drug_b)

        prr, ror, ic = self._compute_disproportionality(
            combo_count, drug_a_count, drug_b_count
        )

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "report_count": combo_count,
            "top_reactions": top_reactions,
            "prr": prr,
            "ror": ror,
            "ic": ic,
            "serious_count": serious,
            "fatal_count": fatal,
        }

    async def query_regimen(self, drugs: List[str],
                             limit: int = _DEFAULT_LIMIT) -> Dict[str, Any]:
        """Return overall adverse event stats for a multi-drug regimen.

        Queries FAERS for reports that mention ALL drugs simultaneously.
        For large regimens (>4 drugs) FAERS specificity drops sharply; in
        that case the coordinator falls back to pairwise queries.

        Returns a dict with the same structure as query_drug_pair plus
        "regimen_drugs": List[str].
        """
        if len(drugs) > 4:
            return await self._pairwise_fallback(drugs, limit)

        parts = [
            f'patient.drug.openfda.generic_name:"{d.lower()}"'
            for d in drugs
        ]
        search = "+AND+".join(parts)
        data = await self._fetch(search=search, limit=limit)
        total = data.get("meta", {}).get("results", {}).get("total", 0)
        top_reactions = self._extract_top_reactions(data)

        # Rough ADR rate estimate: reports / FAERS total
        adr_rate = min(total / max(await self._get_drug_count(drugs[0]), 1), 1.0)

        return {
            "regimen_drugs": drugs,
            "report_count": total,
            "top_reactions": top_reactions,
            "adr_rate_estimate": adr_rate,
        }

    async def get_top_reactions(self, drug: str, limit: int = 20) -> List[Dict]:
        """Return the top adverse event terms for a single drug."""
        search = f'patient.drug.openfda.generic_name:"{drug.lower()}"'
        count_url = f"{_BASE_URL}?search={search}&count=patient.reaction.reactionmeddrapt.exact&limit={limit}"
        if self.api_key:
            count_url += f"&api_key={self.api_key}"
        data = await self._raw_get(count_url)
        results = data.get("results", [])
        return [{"term": r.get("term", ""), "count": r.get("count", 0)} for r in results]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch(self, search: str, limit: int = _DEFAULT_LIMIT) -> Dict:
        params: Dict[str, Any] = {"search": search, "limit": limit}
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{_BASE_URL}?{urlencode(params)}"
        return await self._raw_get(url)

    async def _raw_get(self, url: str) -> Dict:
        """Rate-limited GET with exponential backoff on 429 / 5xx."""
        # Enforce minimum interval between requests
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_request_time)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_time = time.monotonic()

        session = self._session or aiohttp.ClientSession()
        close_after = self._session is None
        try:
            for attempt in range(4):
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status == 404:
                        # No results — OpenFDA returns 404 for zero-count queries
                        return {"meta": {"results": {"total": 0}}, "results": []}
                    if resp.status in (429, 500, 502, 503, 504):
                        delay = 2 ** attempt
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
        finally:
            if close_after:
                await session.close()
        return {"meta": {"results": {"total": 0}}, "results": []}

    async def _get_drug_count(self, drug: str) -> int:
        search = f'patient.drug.openfda.generic_name:"{drug.lower()}"'
        data = await self._fetch(search=search, limit=1)
        return data.get("meta", {}).get("results", {}).get("total", 0)

    async def _count_serious_fatal(self, drug_a: str, drug_b: str) -> Tuple[int, int]:
        base = (
            f'patient.drug.openfda.generic_name:"{drug_a.lower()}"'
            f'+AND+patient.drug.openfda.generic_name:"{drug_b.lower()}"'
        )
        serious_search = base + "+AND+serious:1"
        fatal_search = base + "+AND+seriousnesslifethreatening:1"

        s_data = await self._fetch(search=serious_search, limit=1)
        f_data = await self._fetch(search=fatal_search, limit=1)
        serious = s_data.get("meta", {}).get("results", {}).get("total", 0)
        fatal = f_data.get("meta", {}).get("results", {}).get("total", 0)
        return serious, fatal

    @staticmethod
    def _extract_top_reactions(data: Dict) -> List[Dict]:
        """Extract reaction terms from full event results (not a count query)."""
        reactions: Dict[str, int] = {}
        for report in data.get("results", []):
            for rxn in report.get("patient", {}).get("reaction", []):
                term = rxn.get("reactionmeddrapt", "")
                if term:
                    reactions[term] = reactions.get(term, 0) + 1
        sorted_rxns = sorted(reactions.items(), key=lambda x: x[1], reverse=True)
        return [{"term": t, "count": c} for t, c in sorted_rxns[:10]]

    @staticmethod
    def _compute_disproportionality(n_ab: int, n_a: int, n_b: int
                                    ) -> Tuple[float, float, float]:
        """Compute PRR, ROR, IC for a drug-event / drug-drug pair.

        All three are standard pharmacovigilance disproportionality measures.

        Args:
            n_ab: Reports mentioning both drug A and drug B (or event).
            n_a:  Reports mentioning drug A.
            n_b:  Reports mentioning drug B.

        Returns:
            (PRR, ROR, IC)  — all NaN-safe (returns 0.0 on division errors).
        """
        N = _FAERS_TOTAL_REPORTS
        if n_a == 0 or n_b == 0 or n_ab == 0:
            return 0.0, 0.0, 0.0

        # PRR = (n_ab / n_a) / ((n_b - n_ab) / (N - n_a))
        expected = (n_b / N) * n_a
        prr = n_ab / max(expected, 1e-9)

        # ROR  = (n_ab / (n_a - n_ab)) / ((n_b - n_ab) / (N - n_a - n_b + n_ab))
        try:
            ror_num = n_ab / max(n_a - n_ab, 1)
            ror_den = max(n_b - n_ab, 1) / max(N - n_a - n_b + n_ab, 1)
            ror = ror_num / max(ror_den, 1e-9)
        except ZeroDivisionError:
            ror = 0.0

        # IC (Information Component) = log2(n_ab * N / (n_a * n_b))
        try:
            ic = math.log2(max(n_ab * N / max(n_a * n_b, 1), 1e-9))
        except (ValueError, ZeroDivisionError):
            ic = 0.0

        return round(prr, 3), round(ror, 3), round(ic, 3)

    async def _pairwise_fallback(self, drugs: List[str],
                                  limit: int) -> Dict[str, Any]:
        """For large regimens, run pairwise FAERS queries and aggregate."""
        from itertools import combinations
        results = []
        for drug_a, drug_b in combinations(drugs, 2):
            r = await self.query_drug_pair(drug_a, drug_b, limit)
            results.append(r)

        total_reports = sum(r["report_count"] for r in results)
        all_reactions: Dict[str, int] = {}
        for r in results:
            for rxn in r.get("top_reactions", []):
                all_reactions[rxn["term"]] = (
                    all_reactions.get(rxn["term"], 0) + rxn["count"]
                )
        top_reactions = sorted(all_reactions.items(), key=lambda x: x[1], reverse=True)

        adr_rate = min(total_reports / max(_FAERS_TOTAL_REPORTS, 1), 1.0)
        return {
            "regimen_drugs": drugs,
            "report_count": total_reports,
            "top_reactions": [{"term": t, "count": c} for t, c in top_reactions[:10]],
            "adr_rate_estimate": adr_rate,
            "pairwise_details": results,
        }
