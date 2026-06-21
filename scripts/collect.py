"""Fetch ~250 public top-level comments from r/formula1 into data/raw_comments.csv.

Thread priority (per CLAUDE.md):
  1. Post-race discussion threads   — balanced mix of all three label classes
  2. Race-day live threads          — heavy on `reaction`, some `hot_take`
  3. Daily discussion thread        — general mix
  4. Technical / strategy threads   — deliberately over-sampled to fill `analysis`

`analysis` is rare in comment sections so we hit technical threads last to
top up the quota rather than relying on random collection.
"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "raw_comments.csv"

TARGET_TOTAL = 250
MIN_WARN = 200
MIN_TEXT_LEN = 20  # skip one-word non-signal comments like "lol" or "yes"

DELETED = {"[deleted]", "[removed]"}

# (search query, sort, limit, source_type label, comment quota per thread)
THREAD_SPECS = [
    ("Post Race Discussion",  "new", 4, "race_thread",     40),
    ("Race Thread",           "new", 3, "race_thread",     30),
    ("Daily Discussion",      "new", 2, "daily",           25),
    ("flair:Technical",       "new", 5, "strategy_thread", 35),
    ("strategy analysis",     "new", 5, "strategy_thread", 30),
    ("technical breakdown",   "new", 4, "strategy_thread", 25),
]


def _collect_from_submission(submission, source_type: str, per_thread_limit: int) -> list[dict]:
    import praw.models
    submission.comments.replace_more(limit=0)
    rows = []
    for comment in submission.comments:
        if not isinstance(comment, praw.models.Comment):
            continue
        text = comment.body
        if text in DELETED or len(text) < MIN_TEXT_LEN:
            continue
        rows.append({
            "comment_id": comment.id,
            "text": text,
            "thread_title": submission.title,
            "thread_flair": submission.link_flair_text or "",
            "source_type": source_type,
        })
        if len(rows) >= per_thread_limit:
            break
    return rows


def main() -> None:
    import praw
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()

    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, user_agent]):
        sys.exit(
            "ERROR: set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env"
        )

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        read_only=True,
    )

    subreddit = reddit.subreddit("formula1")
    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    for query, sort, n_threads, source_type, per_thread in THREAD_SPECS:
        print(f"  Fetching '{query}' threads (limit={n_threads}, {per_thread} comments each)…")
        for submission in subreddit.search(query, sort=sort, limit=n_threads, time_filter="month"):
            rows = _collect_from_submission(submission, source_type, per_thread)
            new = [r for r in rows if r["comment_id"] not in seen_ids]
            seen_ids.update(r["comment_id"] for r in new)
            all_rows.extend(new)
            print(f"    [{source_type}] '{submission.title[:60]}' → {len(new)} comments")

    df = pd.DataFrame(all_rows)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    total = len(df)
    print(f"\nCollected {total} unique comments → {OUT_PATH}")

    source_dist = df["source_type"].value_counts()
    print("\nBy source type:")
    for src, count in source_dist.items():
        print(f"  {src}: {count}")

    if total < MIN_WARN:
        print(f"\nWARNING: only {total} candidates (target ≥ {MIN_WARN}). "
              "Consider expanding search queries or increasing thread limits.")


if __name__ == "__main__":
    main()
