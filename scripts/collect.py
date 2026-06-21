"""Fetch ~250 HN comments from Algolia into data/raw_posts.csv.

Six topic queries at even quota (~45 each) keep all three label classes
distributed across domains and prevent any single topic from dominating.
Dedup on objectID across all queries before writing.
"""
import html as html_mod
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "raw_posts.csv"

ALGOLIA_URL = "https://hn.algolia.com/api/v1/search"
QUOTA_PER_QUERY = 45

# Five domains from CLAUDE.md: AI/agentic, programming, infra/startups, quantum, space.
# Six queries so no single topic dominates after dedup.
QUERIES = [
    "llm agents",
    "rust borrow checker",
    "kubernetes",
    "quantum error correction",
    "startup failure",
    "rocket engine",
]

_HIRING_RE = re.compile(r"^Ask HN: Who is hiring", re.IGNORECASE)


def strip_html(raw: str) -> str:
    """Strip HTML tags and unescape entities; preserve casing and punctuation."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html_mod.unescape(text)
    # Collapse runs of whitespace introduced by removed tags
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def fetch_query(query: str, session, limit: int = 100) -> list[dict]:
    """Return up to `limit` non-hiring comment rows for one query."""
    resp = session.get(
        ALGOLIA_URL,
        params={"query": query, "tags": "comment", "hitsPerPage": limit},
        timeout=15,
    )
    resp.raise_for_status()
    rows = []
    for h in resp.json().get("hits", []):
        story_title = h.get("story_title") or ""
        if _HIRING_RE.match(story_title):
            continue
        raw_text = h.get("comment_text") or ""
        text = strip_html(raw_text)
        if not text:
            continue
        rows.append({
            "text": text,
            "story_title": story_title,
            "author": h.get("author", ""),
            "objectID": h.get("objectID", ""),
            "created_at": h.get("created_at", ""),
        })
    return rows


def main() -> None:
    import requests
    import pandas as pd

    session = requests.Session()
    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    print("Fetching HN comments via Algolia…\n")

    per_query: dict[str, int] = {}
    for query in QUERIES:
        candidates = fetch_query(query, session, limit=100)
        fresh = [r for r in candidates if r["objectID"] not in seen_ids]
        kept = fresh[:QUOTA_PER_QUERY]
        seen_ids.update(r["objectID"] for r in kept)
        all_rows.extend(kept)
        per_query[query] = len(kept)
        dupes = len(candidates) - len(fresh)
        print(
            f"  {query!r:30s} → {len(kept):3d} kept"
            f"  (fetched {len(candidates)}, {dupes} cross-query dupes dropped)"
        )

    df = pd.DataFrame(
        all_rows,
        columns=["text", "story_title", "author", "objectID", "created_at"],
    )
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nTotal: {len(df)} unique candidates → {OUT_PATH}")
    if len(df) < 200:
        print(
            f"\nWARNING: only {len(df)} candidates (target ≥ 200). "
            "Increase QUOTA_PER_QUERY or add more queries."
        )


if __name__ == "__main__":
    main()
