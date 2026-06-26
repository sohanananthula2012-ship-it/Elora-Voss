"""
Elara Voss — The Celestial Scribe

A pure LangChain Python agent that searches the web via DuckDuckGo,
synthesizes recent discovery in astronomy, physics, mathematics and
space exploration, and writes elegant scientific journal entries using a
local Ollama LLM.

Usage:
    pip install -r requirements.txt
    # Make sure Ollama is running with a lightweight model, e.g.
    #   ollama run llama3.2
    python elara_voss.py "latest black hole discovery"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("ELARA_MODEL", "llama3.2")
ARCHIVE_DIR = Path(os.getenv("ELARA_ARCHIVE_DIR", "archive"))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PERSONA = """You are Elara Voss, a curious, humble, scholarly, and poetic scientific archivist.
Your domain is astronomy, space exploration, physics, and mathematics.
You speak with scientific accuracy, literary elegance, and a sense of wonder.
You treat every discovery as an invitation, not a conclusion, and you openly
discuss uncertainty and unanswered questions.

You are also fiercely source-based. You ONLY state facts that appear in the
provided search results. You never invent discoveries, quotes, dates, institutions,
or numbers. If the provided sources do not support a claim, you say "uncertain"
or omit it. Every factual claim you make must be traceable to one of the provided
sources, and you cite that source inline."""

FACT_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PERSONA),
    ("human", """You have received a set of web search results about a recent scientific topic.
Your job is to act as a fact-filter. Extract only factual claims that are directly
supported by the search results.

For each claim, return a JSON object with exactly this shape:
{{
  "facts": [
    {{
      "claim": "A concise factual statement",
      "sources": ["Title of source 1 (URL)", "Title of source 2 (URL)"],
      "confidence": "confirmed" | "reported" | "uncertain"
    }}
  ]
}}

Rules:
1. Ignore sensationalized, speculative, or unsupported statements.
2. Only include claims that the search results actually support.
3. If a claim is weakly supported, mark it "uncertain" and include the closest source anyway.
4. Preserve the exact source title and URL for citation.
5. Do not invent facts, institutions, dates, or numbers.
6. Return ONLY the JSON object, with no markdown formatting.

Search results:
{results}
"""),
])

RANKING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PERSONA),
    ("human", """You have just received a set of web search results and an extracted fact list
about a recent scientific topic. Your job is to act as a verification and ranking layer.

Instructions:
1. Read the search results and extracted facts carefully.
2. Select the single most significant, credible, and inspiring discovery.
3. Verify it by cross-referencing the sources shown in the results. Prefer peer-reviewed research, major institutions (NASA, ESA, arXiv, CERN, etc.), and reputable science reporting.
4. Identify uncertainty levels and distinguish confirmed findings from hypotheses.
5. Extract the key mathematical or physical implications.
6. Base every claim in your output on the provided search results and extracted facts.

Return ONLY a JSON object with no markdown formatting, exactly in this shape:
{{
  "title": "Concise, evocative title of the discovery",
  "source_summary": "2-3 sentences summarizing the most credible sources and what they agree on",
  "uncertainty": "low/medium/high — with a brief explanation",
  "confirmed": true or false,
  "key_insight": "The single most important scientific takeaway",
  "larger_questions": "What larger questions does this connect to?",
  "remaining_mysteries": "What remains unresolved?",
  "evidence": [
    {{"claim": "fact from the extracted facts", "sources": ["Title (URL)"], "confidence": "confirmed/report/uncertain"}}
  ]
}}

Extracted facts:
{facts}

Search results:
{results}
"""),
])

JOURNAL_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PERSONA),
    ("human", """Write a self-contained scientific journal entry about the discovery below.

ONLY use facts from the "Search results" and "Extracted facts" sections below.
Do not invent discoveries, quotes, dates, institutions, or numbers.
If a claim is not supported by the provided sources, write "uncertain" or omit it.
Cite sources inline using the format [Source: Title](URL).

Discovery to cover:
{discovery}

Extracted facts (with sources):
{facts}

Original search results:
{results}

Desired length: {length} (short ≈ 500–800 words, standard ≈ 1,000–1,500 words, deep ≈ 2,000–3,000 words)

Structure the entry with the following sections, using the exact headings shown:

Date
Title
Discovery Snapshot
Deeper Context
Mathematical or Physical Significance
Reflective Passage
Visual Inspiration
Closing Thought
Sources

The Sources section must list every source you cited, in the format:
- [Source Title](URL)

Tone: scientifically accurate, literarily elegant, humble about uncertainty, and infused with human curiosity and gentle wonder.

Do not write a preface or apology. Begin directly with the "Date" heading.
"""),
])

LENGTH_WORDS = {
    "short": "500–800 words",
    "standard": "1,000–1,500 words",
    "deep": "2,000–3,000 words",
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ElaraVoss:
    """Autonomous research and content generation agent."""

    def __init__(self, model: str = DEFAULT_MODEL, verbose: bool = True) -> None:
        self.model = ChatOllama(model=model, temperature=0.7)
        self.search_wrapper = DuckDuckGoSearchAPIWrapper()
        self.verbose = verbose
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    def _speak(self, message: str) -> None:
        if self.verbose:
            print(f"[Elara] {message}")

    def search(self, query: str, num_results: int = 10) -> list[dict]:
        """Run a DuckDuckGo search and return structured results."""
        self._speak(f"Searching the web for: {query}")
        try:
            return self.search_wrapper.results(query, num_results)
        except Exception as exc:  # pragma: no cover - search can fail for network reasons
            raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc

    def extract_facts(self, results: list[dict]) -> list[dict]:
        """Run the fact-filter step: extract supported claims from search results."""
        self._speak("Filtering facts from search results...")
        results_text = self._format_results(results)
        chain = FACT_EXTRACTION_PROMPT | self.model
        response = chain.invoke({"results": results_text})
        content = response.content if hasattr(response, "content") else str(response)
        parsed = self._parse_json_like(content)
        facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        # Keep only facts with at least one source.
        return [f for f in facts if isinstance(f, dict) and f.get("sources")]

    def rank_and_verify(self, results: list[dict], facts: list[dict]) -> dict:
        """Ask the local LLM to pick the best discovery and verify it."""
        self._speak("Cross-referencing sources and selecting the most significant discovery...")
        results_text = self._format_results(results)
        facts_text = self._format_facts(facts)
        chain = RANKING_PROMPT | self.model
        response = chain.invoke({"results": results_text, "facts": facts_text})
        content = response.content if hasattr(response, "content") else str(response)
        return self._parse_json_like(content)

    @staticmethod
    def _format_results(results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "N/A")
            link = r.get("link", "N/A")
            snippet = r.get("snippet", r.get("result", "N/A"))
            lines.append(f"[{i}] Title: {title}\n    Link: {link}\n    Snippet: {snippet}")
        return "\n\n".join(lines)

    @staticmethod
    def _format_facts(facts: list[dict]) -> str:
        lines = []
        for i, f in enumerate(facts, start=1):
            claim = f.get("claim", "")
            sources = f.get("sources", [])
            confidence = f.get("confidence", "uncertain")
            sources_text = "; ".join(str(s) for s in sources) if sources else "No source"
            lines.append(f"[{i}] Claim: {claim}\n    Confidence: {confidence}\n    Sources: {sources_text}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_json_like(content: str) -> dict:
        """Best-effort parse of a JSON-like block from the LLM."""
        # Strip markdown fences and surrounding whitespace.
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # If the model didn't give valid JSON, return the raw text as a note.
            return {
                "title": "Unnamed discovery",
                "source_summary": content,
                "uncertainty": "unknown",
                "confirmed": False,
                "key_insight": content,
                "larger_questions": "",
                "remaining_mysteries": "",
            }

    def write_journal(self, discovery: dict, facts: list[dict], results: list[dict], length: str = "standard") -> str:
        """Generate the final journal entry."""
        self._speak("Synthesizing the discovery into a journal entry...")
        length_label = LENGTH_WORDS.get(length, "standard")
        chain = JOURNAL_PROMPT | self.model
        response = chain.invoke({
            "discovery": discovery,
            "facts": self._format_facts(facts),
            "results": self._format_results(results),
            "length": length_label,
        })
        content = response.content if hasattr(response, "content") else str(response)
        return content

    def archive(self, entry: str, title: str) -> Path:
        """Save a journal entry to the local archive."""
        today = datetime.now().strftime("%Y-%m-%d")
        slug = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:60] or "discovery"
        filename = f"{today}_{slug}.md"
        path = ARCHIVE_DIR / filename
        path.write_text(entry, encoding="utf-8")
        self._speak(f"Archived entry to {path}")
        return path

    def run(self, query: Optional[str] = None, length: str = "standard", num_results: int = 10) -> Path:
        """Execute the full daily workflow."""
        if query is None:
            query = "recent astronomy discoveries this week"

        results = self.search(query, num_results=num_results)
        if not results:
            raise RuntimeError("No search results returned. Check your network connection.")

        facts = self.extract_facts(results)
        discovery = self.rank_and_verify(results, facts)
        discovery.setdefault("_query", query)

        # Add a date field if the model did not provide one.
        discovery["date"] = datetime.now().strftime("%B %d, %Y")

        entry = self.write_journal(discovery, facts, results, length=length)

        # Ensure the entry starts with a Date heading.
        if not entry.lstrip().startswith("Date"):
            entry = f"Date: {discovery['date']}\n\n{entry}"

        # Ensure a Sources section exists.
        entry = self._ensure_sources_section(entry, facts, results)

        title = discovery.get("title", "Untitled Discovery")
        return self.archive(entry, title)

    @staticmethod
    def _ensure_sources_section(entry: str, facts: list[dict], results: list[dict]) -> str:
        """Append a Sources section if the journal does not already contain one."""
        if "Sources" in entry:
            return entry

        # Build a deduplicated source list from facts and results.
        seen: set[str] = set()
        source_lines: list[str] = []
        for f in facts:
            for s in f.get("sources", []):
                match = re.search(r"\((https?://[^\)]+)\)", str(s))
                if match:
                    url = match.group(1)
                    title = s[: match.start()].strip().rstrip(" (").strip()
                else:
                    url = s
                    title = s
                if url not in seen:
                    seen.add(url)
                    source_lines.append(f"- [{title}]({url})")
        for r in results:
            url = r.get("link")
            title = r.get("title", "Untitled source")
            if url and url not in seen:
                seen.add(url)
                source_lines.append(f"- [{title}]({url})")

        if source_lines:
            entry += "\n\n## Sources\n\n" + "\n".join(source_lines)
        return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Elara Voss — The Celestial Scribe")
    parser.add_argument("query", nargs="?", help="Search query (defaults to recent astronomy discoveries)")
    parser.add_argument(
        "--length",
        choices=["short", "standard", "deep"],
        default="standard",
        help="Journal length (default: standard)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--results", type=int, default=10, help="Number of search results to fetch")
    args = parser.parse_args()

    agent = ElaraVoss(model=args.model)
    output_path = agent.run(query=args.query, length=args.length, num_results=args.results)
    print(f"\nJournal saved to: {output_path}")


if __name__ == "__main__":
    main()
