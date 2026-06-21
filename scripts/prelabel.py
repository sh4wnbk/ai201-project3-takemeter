"""Pre-label raw_posts.csv using Cerebras llama-3.3-70b.

label_suggested is write-once: rows already in working_annotations.csv are
never re-labeled, preserving the draft history for disclosure.
story_title is carried through as a read-only context column for human review;
it is not passed to the model.
"""
import os
import re
import sys
from pathlib import Path
from typing import Optional

VALID_LABELS = {"analysis", "hot_take", "reaction"}

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_PATH = DATA_DIR / "raw_posts.csv"
WORK_PATH = DATA_DIR / "working_annotations.csv"

CLASSIFICATION_PROMPT = """\
You are a text classifier. Classify the following Hacker News comment into exactly one of:
  analysis, hot_take, reaction

Definitions:
- analysis: a structured claim backed by specific, verifiable evidence (a mechanism, \
benchmark, technical reasoning, or citation). Strip the opinion framing — the evidence \
still stands on its own as an argument. \
Example: "The latency win isn't the Rust rewrite — they moved the hot path off the GC'd \
heap; you'd get the same gain in Go with a sync.Pool."
- hot_take: a bold, confident opinion asserted without real evidence. Technical vocabulary \
alone does NOT make a comment analysis. If removing the opinion framing leaves nothing \
substantive — it is hot_take. \
Example: "Kubernetes is wildly over-engineered for 99% of companies. Just use a VM."
- reaction: emotion- or humor-driven; short quips, jokes, praise, snark. No argued claim. \
Example: "This is the most beautiful codebase I've seen all year, wow."

Decision rules:
1. hot_take vs analysis: if removing the opinion framing leaves genuine supporting \
evidence → analysis. If the evidence is a decorative technical noun or one cherry-picked \
detail with no argument behind it → hot_take. \
   Example trap: "Their architecture fundamentally can't scale, the event loop blocks on \
I/O, full stop" → hot_take (names real concepts, shows no mechanism).
2. reaction vs hot_take (sarcasm): if the primary act is asserting a position, even \
sarcastically → hot_take. If the primary act is the joke/bit itself with no real \
position under it → reaction.

Respond with ONLY the label name and nothing else. No explanation, no punctuation.

Comment: {text}"""


def parse_label(raw_response: str) -> Optional[str]:
    """Map a raw LLM response string to one of the three valid labels.

    Returns None for empty, ambiguous, or unparseable input rather than
    silently assigning a wrong label.
    """
    if not raw_response or not raw_response.strip():
        return None

    # Normalise: strip whitespace, lowercase, strip trailing punctuation
    cleaned = raw_response.strip().lower()
    cleaned = re.sub(r"[.!?,;:\-]+$", "", cleaned).strip()

    # Normalise hyphen variant before matching
    cleaned = cleaned.replace("hot-take", "hot_take")

    # Exact match (fastest path, handles the clean case)
    if cleaned in VALID_LABELS:
        return cleaned

    # Substring search — find every valid label present
    found = [label for label in VALID_LABELS if label in cleaned]

    if len(found) == 1:
        return found[0]

    # Zero or multiple matches — ambiguous or garbage
    return None


def _build_prompt(text: str) -> str:
    return CLASSIFICATION_PROMPT.format(text=text)


def _call_cerebras(text: str, client) -> Optional[str]:
    response = client.chat.completions.create(
        model="llama-3.3-70b",
        messages=[{"role": "user", "content": _build_prompt(text)}],
        temperature=0,
        max_tokens=10,
    )
    raw = response.choices[0].message.content or ""
    return parse_label(raw)


def main() -> None:
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        sys.exit("ERROR: CEREBRAS_API_KEY not set in .env")

    if not RAW_PATH.exists():
        sys.exit(f"ERROR: {RAW_PATH} not found — run collect.py first")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")

    raw_df = pd.read_csv(RAW_PATH, dtype=str)

    # Load existing work file to respect write-once on label_suggested
    already_labeled_ids: set = set()
    existing_df: Optional[pd.DataFrame] = None
    if WORK_PATH.exists():
        existing_df = pd.read_csv(WORK_PATH, dtype=str)
        already_labeled_ids = set(
            existing_df.loc[existing_df["label_suggested"].notna(), "objectID"]
        )
        print(f"Resuming: {len(already_labeled_ids)} rows already pre-labeled, skipping.")

    to_process = raw_df[~raw_df["objectID"].isin(already_labeled_ids)].copy()
    print(f"Pre-labeling {len(to_process)} new rows via Cerebras…")

    new_rows = []
    unknown_count = 0
    for _, row in to_process.iterrows():
        text = row["text"]
        label_suggested = _call_cerebras(text, client)
        if label_suggested is None:
            label_suggested = "unknown"
            unknown_count += 1
            print(f"  WARNING: unparseable response for objectID={row['objectID']}")
        new_rows.append({
            "objectID": row["objectID"],
            "text": text,
            "story_title": row.get("story_title", ""),
            "label_suggested": label_suggested,
            "label": "",
            "prelabeled": True,
            "notes": "",
        })

    new_df = pd.DataFrame(new_rows)

    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(WORK_PATH, index=False)
    print(f"\nWrote {len(combined)} rows to {WORK_PATH}")
    if unknown_count:
        print(f"  {unknown_count} rows have label_suggested='unknown' — review manually.")

    # Distribution report
    dist = combined["label_suggested"].value_counts()
    total = len(combined)
    print("\nlabel_suggested distribution:")
    for label, count in dist.items():
        pct = count / total * 100
        flag = "  *** >70% — rebalance before reviewing ***" if pct > 70 else ""
        print(f"  {label}: {count} ({pct:.1f}%){flag}")


if __name__ == "__main__":
    main()
