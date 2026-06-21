# CLAUDE.md — TakeMeter

Context for Claude Code. Read this fully before doing anything. This is a CodePath AI 201 project: a fine-tuned text classifier that sorts r/formula1 comments by discourse type. Your job in this phase is the **data pipeline and repo scaffolding** — not the model training (that happens in a Colab notebook) and not the final labeling decisions (the human reviews every label).

---

## Locked decisions — do not change these

**Community:** r/formula1.

**Labels (exactly three, single axis of analytical substance):**

- `analysis` — structured claim backed by specific, verifiable evidence (tactics, strategy, stats, regs, a technical mechanism). Strip the opinion and the evidence still stands.
- `hot_take` — bold, confident opinion asserted *without* real evidence; may cite one decorative stat selected for effect, not argument.
- `reaction` — non-analytical, emotion- or humor-driven: in-the-moment feeling, memes, banter, copypasta. No argued claim.

**Decision rules for the two hard boundaries (use these in any pre-labeling prompt):**

1. `hot_take` vs `analysis` — **technical vocabulary alone does NOT make a post `analysis`.** Look for an actual mechanism or verifiable evidence. If removing the opinion framing leaves something that genuinely supports the claim → `analysis`. If the "evidence" is a decorative technical noun or one cherry-picked stat with no argument → `hot_take`. (Example trap: *"Mercedes' floor concept is fundamentally wrong, full stop"* → `hot_take`.)
2. `reaction` vs `hot_take` (sarcasm) — if the post's primary act is asserting a position, even sarcastically → `hot_take`. If its primary act is the joke/bit with no real position under it → `reaction`.

**Success threshold (for later, the model must hit):** macro-F1 ≥ 0.70, no single class F1 < 0.60, beat the zero-shot baseline by ≥ 10 points.

**Baseline provider:** Cerebras, not Groq (Groq signup has errored across all projects). Model `llama-3.3-70b`, OpenAI-compatible endpoint `https://api.cerebras.ai/v1`, key in `.env` as `CEREBRAS_API_KEY`. The same client is reused for pre-labeling.

---

## The data pipeline (your main deliverable)

Three stages. Stage 3 is the human's, not yours.

1. **Collect** (`scripts/collect.py`) — pull public top-level comments from r/formula1 into `data/raw_comments.csv`. Target ~250 candidates so there's slack to drop junk and rebalance. Source threads, in priority order: post-race discussion threads, race-day live threads, the daily discussion thread, and — specifically to fill the `analysis` quota — posts with technical/strategy flair and strategy-breakdown threads. `analysis` is the rarest class in comment sections, so deliberately over-sample it; do not collect purely at random.

2. **Pre-label** (`scripts/prelabel.py`) — run each raw comment through Cerebras `llama-3.3-70b` (temperature 0) with a prompt built from the label definitions and decision rules above. Write the draft to a `label_suggested` column and set `prelabeled=True`. Output to `data/working_annotations.csv` with columns: `text`, `label_suggested`, `label` (blank — human fills), `prelabeled`, `notes`.

3. **Human review** (the human does this, not you) — the human reads every row, fills the real `label`, and corrects `label_suggested` where wrong. A small export step then writes the final `data/takemeter_dataset.csv` with exactly three columns: `text`, `label`, `notes`. Build that export script (`scripts/export_dataset.py`) but never populate `label` yourself.

---

## Hard rules / guardrails

- **Never produce the final labeled dataset.** You may generate `label_suggested` drafts only. The `label` column is the human's. Do not copy `label_suggested` into `label`.
- **Preserve text exactly** — keep emoji, casing ("CRASHHHH"), and punctuation. They are real signal for `reaction`. Do not lowercase, strip emoji, or "clean" the text.
- **Balance:** after collection, print the candidate count and, after pre-labeling, the `label_suggested` distribution. Flag if any class looks likely to exceed 70% so the human can rebalance before reviewing.
- **Secrets:** `.env` is gitignored and never committed. Reddit and Cerebras credentials live there only.
- **Disclosure:** pre-labeling is AI assistance that must be disclosed in the README. Keep `prelabeled` in the working file so the human can report how many rows were pre-labeled and how many they corrected.
- **Don't over-build collection.** A focused PRAW fetch script is fine; do not turn this into a scraping framework.

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
│   ├── raw_comments.csv        # collect.py output
│   ├── working_annotations.csv # prelabel.py output; human reviews here
│   └── takemeter_dataset.csv   # final: text,label,notes (export step)
├── scripts/
│   ├── collect.py
│   ├── prelabel.py
│   └── export_dataset.py
└── app/
    └── interface.py            # Gradio stretch feature (built later)
```

## Stack / environment

- **Local scripts:** Python, `praw` (Reddit), `openai` (Cerebras client), `python-dotenv`, `pandas`. These go in `requirements.txt`.
- **Model training + eval:** happens in a separate **Google Colab** notebook (`distilbert-base-uncased`, T4 GPU, 3 epochs, lr 2e-5, batch 16, 70/15/15 split). Do **not** add transformers/torch training code to this repo — it lives in Colab. The notebook also runs the Cerebras zero-shot baseline (Section 5).
- **`.env` keys:** `CEREBRAS_API_KEY`, plus Reddit `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` for PRAW.

## Stretch features committed (later phases)

- **Error pattern analysis** — name the technical-vocabulary trap as a systematic pattern with evidence from the error set.
- **Confidence calibration** — bin test predictions by softmax confidence, check accuracy per bin (report as directional given ~30 test examples).
- **Deployed interface** — `app/interface.py`, a Gradio app: post in → label + confidence out. Reuses the Project 2 (FitFindr) Gradio pattern.

## Definition of done for THIS handoff

Repo scaffolded; `.gitignore` + `requirements.txt` in place; `collect.py`, `prelabel.py`, `export_dataset.py` written and runnable; `data/` schemas created. Collection and pre-labeling can be *run* once the human provides credentials. The final `label` column is left empty for the human.
