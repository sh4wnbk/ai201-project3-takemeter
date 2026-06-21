# CLAUDE.md — TakeMeter

Context for Claude Code. Read this fully before doing anything. This is a CodePath AI 201 project: a fine-tuned text classifier that sorts Hacker News comments by discourse type (analysis / hot_take / reaction). Your job in this phase is the **data pipeline and repo scaffolding** — not the model training (that happens in a Colab notebook) and not the final labeling decisions (the human reviews every label).

---

## Locked decisions — do not change these

**Community:** Hacker News (comments, weighted toward AI / programming / startups / quantum / space threads).

**Labels (exactly three, single axis of analytical substance):**

- `analysis` — structured claim backed by specific, verifiable evidence (mechanism, benchmarks, technical reasoning, citations). Strip the opinion and the evidence still stands. *e.g. "The latency win isn't the Rust rewrite — they moved the hot path off the GC'd heap; you'd get the same gain in Go with a sync.Pool."*
- `hot_take` — bold, confident opinion asserted *without* real evidence; may name-drop a technical concept for effect, not argument. *e.g. "Kubernetes is wildly over-engineered for 99% of companies. Just use a VM."*
- `reaction` — non-analytical, emotion- or humor-driven: short quips, jokes, praise, snark. No argued claim. *e.g. "This is the most beautiful codebase I've seen all year, wow."*

**Decision rules for the two hard boundaries (use these in any pre-labeling prompt):**

1. `hot_take` vs `analysis` — **technical vocabulary alone does NOT make a post `analysis`.** Look for an actual mechanism or verifiable evidence. If removing the opinion framing leaves something that genuinely supports the claim → `analysis`. If the "evidence" is a decorative technical noun or one cherry-picked detail with no argument → `hot_take`. (Example trap: *"Their architecture fundamentally can't scale, the event loop blocks on I/O, full stop"* → `hot_take`: names real concepts, shows no mechanism.)
2. `reaction` vs `hot_take` (sarcasm) — if the post's primary act is asserting a position, even sarcastically → `hot_take`. If its primary act is the joke/bit with no real position under it → `reaction`.

**Success threshold (for later, the model must hit):** macro-F1 ≥ 0.70, no single class F1 < 0.60, beat the zero-shot baseline by ≥ 10 points.

**Baseline provider:** Cerebras, not Groq (Groq signup has errored across all projects). Model `llama-3.3-70b`, OpenAI-compatible endpoint `https://api.cerebras.ai/v1`, key in `.env` as `CEREBRAS_API_KEY`. The same client is reused for pre-labeling.

---

## The data pipeline (your main deliverable)

Three stages. Stage 3 is the human's, not yours.

1. **Collect** (`scripts/collect.py`) — pull public comments from **Hacker News** into `data/raw_posts.csv`. Target ~250 candidates so there's slack to drop junk and rebalance.

   **SOURCE: Hacker News.** Reddit (Responsible Builder Policy) and YouTube both prohibit using their data to train ML models. Bluesky failed on data quality (no analytical discourse density). HN's API is open (no auth, no keys, no app), and HN is dense, long-form technical discourse — well-suited to the analysis/hot_take/reaction split.

   API: use the **Algolia HN Search API** — `https://hn.algolia.com/api/v1/search?query=<term>&tags=comment` (relevance) or `search_by_date` (recency), via plain `requests`. Each hit already contains `comment_text`, `author`, `story_title`, `parent_id`, `objectID`, `created_at` — no auth, no thread-walking, no Firebase needed. (Firebase `hacker-news.firebaseio.com/v0` is only a fallback if a full comment tree by ID is ever needed.)

   Queries: search topic terms in domains the human knows well so review labeling stays accurate — AI/agentic engineering, programming, startups, quantum, space (e.g. `llm agents`, `rust borrow checker`, `kubernetes`, `quantum error correction`). To fill the `analysis` quota, bias toward debate-heavy technical threads; `analysis` is the rarest class, so do not collect purely at random. Comment length is a weak proxy — keep short comments for `reaction`, longer argued ones for `analysis`.

2. **Pre-label** (`scripts/prelabel.py`) — read `data/raw_posts.csv` and run each post through Cerebras `llama-3.3-70b` (temperature 0) with a prompt built from the label definitions and decision rules above. Write the draft to a `label_suggested` column and set `prelabeled=True`. Output to `data/working_annotations.csv` with columns: `text`, `label_suggested`, `label` (blank — human fills), `prelabeled`, `notes`.

3. **Human review** (the human does this, not you) — the human reads every row, fills the real `label`, and corrects `label_suggested` where wrong. A small export step then writes the final `data/takemeter_dataset.csv` with exactly three columns: `text`, `label`, `notes`. Build that export script (`scripts/export_dataset.py`) but never populate `label` yourself.

---

## Hard rules / guardrails

- **Never produce the final labeled dataset.** You may generate `label_suggested` drafts only. The `label` column is the human's. Do not copy `label_suggested` into `label`.
- **`label_suggested` is write-once.** Once pre-labeling writes a draft, never mutate it. The human's corrections go to the separate `label` column; the draft stays intact. This is what lets the human report how many drafts they overrode for the disclosure — overwriting the draft destroys that history. (This is the same class of bug flagged in a prior review: accumulated state must preserve the full history of decisions, not just the latest one.)
- **Preserve text exactly** — keep emoji, casing ("CRASHHHH"), and punctuation. They are real signal for `reaction`. Do not lowercase, strip emoji, or "clean" the text.
- **Design for offline testability.** Any logic that can be separated from a network call must be. In `prelabel.py`, the response-to-label mapping goes in a pure function — `parse_label(raw_response: str) -> str` — that takes a string and returns one of the three labels (or flags an unparseable response), with no API call inside it. The Cerebras call stays a thin wrapper around it. This makes the parsing unit-testable without hitting the API, which is the highest-leverage habit flagged in prior reviews.
- **Balance:** after collection, print the candidate count and, after pre-labeling, the `label_suggested` distribution. Flag if any class looks likely to exceed 70% so the human can rebalance before reviewing.
- **Secrets:** `.env` is gitignored and never committed. Bluesky (if auth needed) and Cerebras credentials live there only.
- **Disclosure:** pre-labeling is AI assistance that must be disclosed in the README. Keep `prelabeled` in the working file so the human can report how many rows were pre-labeled and how many they corrected.
- **Don't over-build collection.** A focused Bluesky fetch script is fine; do not turn this into a scraping framework.

---

## Repo structure

```
ai201-project3-takemeter/
├── CLAUDE.md                  # this file
├── planning.md                # design doc (human drops in — already written)
├── README.md                  # skeleton now, filled at the end
├── .gitignore                 # .env, __pycache__, .ipynb_checkpoints
├── requirements.txt           # local scripts only (see below)
├── data/
│   ├── raw_posts.csv           # collect.py output (Bluesky posts)
│   ├── working_annotations.csv # prelabel.py output; human reviews here
│   └── takemeter_dataset.csv   # final: text,label,notes (export step)
├── scripts/
│   ├── collect.py
│   ├── prelabel.py
│   └── export_dataset.py
├── tests/
│   ├── test_parse_label.py     # offline: parse_label() normalizes/rejects raw responses
│   └── test_export.py          # offline: export skips unlabeled rows, preserves text exactly
└── app/
    └── interface.py            # Gradio stretch feature (built later)
```

## Stack / environment

- **Local scripts:** Python, `requests` + `atproto` (Bluesky), `openai` (Cerebras client), `python-dotenv`, `pandas`, `pytest`. These go in `requirements.txt`. (`atproto` only needed if the public host requires auth — see Collect.)
- **Model training + eval:** happens in a separate **Google Colab** notebook (`distilbert-base-uncased`, T4 GPU, 3 epochs, lr 2e-5, batch 16, 70/15/15 split). Do **not** add transformers/torch training code to this repo — it lives in Colab. The notebook also runs the Cerebras zero-shot baseline (Section 5).
- **`.env` keys:** `CEREBRAS_API_KEY`, plus `BLUESKY_HANDLE` / `BLUESKY_APP_PASSWORD` **only if** Bluesky post search requires auth (the public host may not).

## Testing (build test-first, not after)

Write these as you build the scripts, not at the end. All must run offline — no network, no credentials.

- `tests/test_parse_label.py` — exercises `parse_label()` from `prelabel.py`: clean labels (`"analysis"`), messy ones that should normalize (`"  Analysis."`, `"the label is hot_take"`), and garbage that should be rejected/flagged rather than silently mislabeled.
- `tests/test_export.py` — exercises `export_dataset.py`: rows with an empty `label` are excluded from the final CSV; `text` is preserved byte-for-byte (emoji and casing intact, no cleaning); output has exactly the three columns `text`, `label`, `notes`.

`pytest` must pass before the handoff is considered done.

## Stretch features committed (later phases)

- **Error pattern analysis** — name the technical-vocabulary trap as a systematic pattern with evidence from the error set.
- **Confidence calibration** — bin test predictions by softmax confidence, check accuracy per bin (report as directional given ~30 test examples).
- **Deployed interface** — `app/interface.py`, a Gradio app: post in → label + confidence out. Reuses the Project 2 (FitFindr) Gradio pattern.

## Definition of done for THIS handoff

Repo scaffolded; `.gitignore` + `requirements.txt` in place; `collect.py`, `prelabel.py` (with `parse_label()` as a pure, network-free function), `export_dataset.py` written and runnable; `data/` schemas created; `tests/test_parse_label.py` and `tests/test_export.py` written and passing under `pytest`. Collection and pre-labeling can be *run* once the human provides credentials. The final `label` column is left empty for the human.
