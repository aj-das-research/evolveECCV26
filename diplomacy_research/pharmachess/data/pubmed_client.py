"""PubMed E-utilities Client

Provides literature-backed evidence for PharmaChess signal assessment.
Uses NCBI's E-utilities API (https://www.ncbi.nlm.nih.gov/books/NBK25500/).

The SentinelAgent (and the PharmaChess LiteratureAgent) calls this to:
  1. Count published case reports / clinical studies for a drug combination.
  2. Retrieve titles + abstracts for mechanistic evidence.
  3. Search for drug safety signals in the recent literature.

Rate limits:
  - Without NCBI API key : 3 requests / second
  - With NCBI API key    : 10 requests / second  (free, register at ncbi.nlm.nih.gov)

Set NCBI_API_KEY in the environment.  Also set NCBI_EMAIL to the email
you registered with (courteous API use per NCBI guidelines).
"""
import asyncio
import os
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, quote_plus

import aiohttp


_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class PubMedClient:
    """Async PubMed E-utilities client for PharmaChess literature reasoning.

    Example:
        async with PubMedClient(api_key=os.environ["NCBI_API_KEY"]) as pm:
            articles = await pm.search_drug_interaction("rivaroxaban", "aspirin")
            print(articles[0]["title"])
    """

    def __init__(self, api_key: Optional[str] = None,
                 email: Optional[str] = None):
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.email = email or os.environ.get("NCBI_EMAIL", "pharmachess@example.com")
        # Respect rate limits: 3 req/s without key, 10 req/s with key
        self._min_interval = 1.0 / (10.0 if self.api_key else 3.0)
        self._last_request_time: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # High-level queries
    # ------------------------------------------------------------------

    async def search_drug_interaction(self, drug_a: str, drug_b: str,
                                       max_results: int = 10) -> List[Dict[str, Any]]:
        """Search PubMed for articles about a specific drug-drug interaction.

        Query strategy:
            (drug_a[tiab] AND drug_b[tiab] AND
             (interaction[tiab] OR adverse[tiab] OR toxicity[tiab]))

        Returns a list of article dicts with: pmid, title, abstract (truncated),
        journal, pub_date, authors.
        """
        query = (
            f'("{drug_a}"[tiab] AND "{drug_b}"[tiab] AND '
            f'("interaction"[tiab] OR "adverse"[tiab] OR '
            f'"toxicity"[tiab] OR "pharmacokinetic"[tiab]))'
        )
        pmids = await self._esearch(query, max_results)
        if not pmids:
            return []
        return await self._efetch_summaries(pmids)

    async def search_adverse_event(self, drug: str, event_term: str,
                                    max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for literature on a specific drug adverse event.

        Args:
            drug:       Generic drug name.
            event_term: MedDRA preferred term or plain description.
        """
        query = f'("{drug}"[tiab] AND "{event_term}"[tiab] AND "case report"[pt])'
        pmids = await self._esearch(query, max_results)
        if not pmids:
            # Broaden search if no case reports found
            query = f'("{drug}"[tiab] AND "{event_term}"[tiab])'
            pmids = await self._esearch(query, max_results)
        return await self._efetch_summaries(pmids) if pmids else []

    async def get_article_count(self, drug_a: str,
                                 drug_b: Optional[str] = None) -> int:
        """Return the total PubMed article count for a drug (pair).

        Used as a literature-evidence weight in signal scoring.
        """
        if drug_b:
            query = f'("{drug_a}"[tiab] AND "{drug_b}"[tiab])'
        else:
            query = f'("{drug_a}"[tiab])'
        pmids = await self._esearch(query, max_results=1)
        # esearch returns total count in metadata; we use len as proxy here
        # For exact counts we parse the Count field from raw XML
        return await self._esearch_count(query)

    async def get_recent_safety_articles(self, drug: str,
                                          years_back: int = 3,
                                          max_results: int = 5) -> List[Dict]:
        """Retrieve recent safety-relevant publications for a drug.

        Useful for the self-evolution loop: newly published case reports
        may update the agent's memory without human annotation.
        """
        from datetime import datetime
        current_year = datetime.now().year
        from_year = current_year - years_back
        query = (
            f'("{drug}"[tiab] AND ("adverse"[tiab] OR "safety"[tiab] OR '
            f'"interaction"[tiab]) AND ("{from_year}"[dp]:"{current_year}"[dp]))'
        )
        pmids = await self._esearch(query, max_results)
        return await self._efetch_summaries(pmids) if pmids else []

    # ------------------------------------------------------------------
    # E-utilities raw calls
    # ------------------------------------------------------------------

    async def _esearch(self, query: str, max_results: int) -> List[str]:
        """Run esearch and return a list of PMIDs."""
        params: Dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        data = await self._get(_ESEARCH_URL, params)
        if not data:
            return []
        try:
            return data["esearchresult"]["idlist"]
        except (KeyError, TypeError):
            return []

    async def _esearch_count(self, query: str) -> int:
        """Return the exact total count from esearch without fetching results."""
        params: Dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": 0,
            "retmode": "json",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        data = await self._get(_ESEARCH_URL, params)
        try:
            return int(data["esearchresult"].get("count", 0))
        except (KeyError, TypeError, ValueError):
            return 0

    async def _efetch_summaries(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch article summaries (title, abstract, metadata) for a list of PMIDs."""
        if not pmids:
            return []
        params: Dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
            "rettype": "summary",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        data = await self._get(_ESUMMARY_URL, params)
        articles = []
        try:
            result = data.get("result", {})
            for pmid in pmids:
                art = result.get(pmid, {})
                if not art:
                    continue
                articles.append({
                    "pmid":     pmid,
                    "title":    art.get("title", ""),
                    "journal":  art.get("source", ""),
                    "pub_date": art.get("pubdate", ""),
                    "authors":  [
                        a.get("name", "") for a in art.get("authors", [])[:3]
                    ],
                    # abstract not in esummary; fetch separately if needed
                    "abstract": "",
                })
        except (KeyError, TypeError):
            pass
        return articles

    async def fetch_abstract(self, pmid: str) -> str:
        """Fetch the full abstract for a single PMID (efetch XML)."""
        params: Dict[str, Any] = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "rettype": "abstract",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        raw = await self._get_text(_EFETCH_URL, params)
        if not raw:
            return ""
        try:
            root = ET.fromstring(raw)
            texts = root.findall(".//AbstractText")
            return " ".join(t.text or "" for t in texts if t.text)
        except ET.ParseError:
            return raw[:500]

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    async def _get(self, url: str, params: Dict) -> Dict:
        text = await self._get_text(url, params)
        if not text:
            return {}
        try:
            import json
            return json.loads(text)
        except Exception:
            return {}

    async def _get_text(self, url: str, params: Dict) -> str:
        await self._rate_limit()
        full_url = f"{url}?{urlencode(params)}"
        session = self._session or aiohttp.ClientSession()
        close_after = self._session is None
        try:
            for attempt in range(4):
                try:
                    async with session.get(
                        full_url, timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            return await resp.text()
                        if resp.status in (429, 500, 502, 503, 504):
                            await asyncio.sleep(2 ** attempt)
                            continue
                except aiohttp.ClientError:
                    await asyncio.sleep(2 ** attempt)
        finally:
            if close_after:
                await session.close()
        return ""

    async def _rate_limit(self):
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_request_time)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_time = time.monotonic()
