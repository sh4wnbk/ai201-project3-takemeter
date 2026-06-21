# TakeMeter — Planning

A fine-tuned classifier that sorts r/formula1 comments by what *kind* of discourse they are: a reasoned argument, a bare opinion, or a gut response. The point isn't to judge whether a take is *right* — it's to separate posts that do analytical work from posts that just assert or react.

---

## 1. Community

**r/formula1.** F1 race threads are an unusually clean fit for this task because the same event produces all three kinds of discourse at once. A single overtake spawns a strategist breaking down tyre deg and undercut windows, a fan declaring a driver "finished," and someone just screaming at the safety car. The community itself polices this distinction — "source?" and "that's not analysis, that's a hot take" are native phrases there. That means the boundary I'm modeling is one regulars already recognize, not one I'm imposing from outside.

It also matches a domain I know well, which matters more than it sounds: the failure mode that kills these projects is *inconsistent annotation*, and I can label F1 discourse consistently because I can actually tell whether a strategy claim holds up or is decorative.

---

## 2. Labels

Three labels on a single axis: descending analytical substance.

**`analysis`** — A post that makes a structured claim backed by specific, verifiable evidence (tactics, strategy, telemetry/stats, regulations, or a technical mechanism). If you strip the opinion framing, the evidence still stands on its own as an argument.
- *"Strat call was insane — pitting under the VSC saved ~11s vs staying out, that's the whole podium."*
- *"He's quicker on the softs but his tyre deg over a stint has been the worst on the grid all season, that's why they go long."*

**`hot_take`** — A bold, confident opinion asserted *without* real evidence. It may cite one decorative stat, but the stat is selected for effect rather than as part of an argument. The claim might even be true — the post just doesn't do the work to show it.
- *"Verstappen is just better than everyone, end of."*
- *"Mercedes hasn't won because their floor concept is fundamentally wrong, full stop."*

**`reaction`** — A non-analytical post driven by emotion or humor: an in-the-moment feeling, a meme, banter, or copypasta. There's no claim being argued — the post is expressing a feeling or making a joke.
- *"CRASHHHH did you SEE that"*
- *"Imagine being a Williams fan in 2026, couldn't be me 💀"*

These work because each boundary can be stated in one sentence, two readers applying the definitions would agree on most posts, and the distinctions reflect how the community actually talks.

---

## 3. Hard edge cases

**Primary edge case — `hot_take` vs. `analysis` (the technical-vocabulary trap).** The dangerous post is one that *sounds* analytical but is structurally a bare assertion: *"Mercedes hasn't won because their floor concept is fundamentally wrong, full stop."* It uses technical vocabulary ("floor concept") but offers no mechanism, no evidence, no reasoning — just a confident verdict.

> **Decision rule:** Technical vocabulary alone does **not** make a post `analysis`. Ask whether the post supplies an actual mechanism or verifiable evidence. If removing the opinion framing leaves something that genuinely supports the claim → `analysis`. If the "evidence" is decorative — one cherry-picked stat or a technical noun with no argument behind it → `hot_take`.

This is also my predicted **model failure mode**: DistilBERT keys heavily on token-level vocabulary, so it will likely over-predict `analysis` on technical words alone. I expect the confusion matrix to show `analysis` ← `hot_take` errors concentrated on technical-sounding assertions, and I'm annotating deliberately so that prediction is testable rather than self-fulfilling.

**Secondary edge case — sarcasm between `hot_take` and `reaction`.** A deadpan sarcastic line ("oh yeah, Ferrari *nailed* that one") carries an implied opinion but reads as a joke.

> **Decision rule:** If the post's primary act is asserting a position (even sarcastically), label `hot_take`. If its primary act is the joke/bit itself with no real position underneath, label `reaction`. Ask: is the author arguing, or performing?

---

## 4. Data collection plan

**Source:** Public r/formula1 comments, collected manually into a spreadsheet — primarily from post-race discussion threads, race-day live threads, and the daily discussion thread. Public comments only; no private channels.

**Target:** ~200 total, aiming for rough balance (~60–70 per label). No single label may exceed 70% of the set.

**The expected imbalance, and the fix:** Comment sections skew heavily toward `reaction` and `hot_take`; genuine `analysis` is rarer and clusters in specific places. So I'll deliberately over-sample `analysis` from technical/strategy threads and posts with technical flair, rather than collecting randomly and hoping. If `analysis` is still underrepresented after a first pass, I collect a second targeted batch from strategy-breakdown threads before annotating further — I will not pad the other classes to compensate, since that just trains a majority-class predictor.

**Format:** One CSV, columns `text`, `label`, `notes` (for difficult cases). The notebook handles the 70/15/15 split — I save one labeled file, not pre-split files.

---

## 5. Evaluation metrics

**Accuracy alone is not enough** because it hides per-class failure. If `analysis` is the rarest class and the model never predicts it, accuracy can still look respectable while the model has completely failed at the one boundary that matters most. So:

- **Macro-F1** as the headline metric — it averages F1 across classes equally, so a collapsed `analysis` class drags the score down instead of hiding behind a large `reaction` class.
- **Per-class precision / recall / F1** — to see *which* class is weak and in which direction (is the model conservative or over-eager on `analysis`?).
- **Confusion matrix** — to read the *direction* of errors. I specifically expect a hot spot at (true=`hot_take`, pred=`analysis`); the matrix confirms or refutes the technical-vocabulary hypothesis from §3.

Together these tell me not just *how often* the model is wrong but *where* and *why* — which is what the error analysis is graded on.

---

## 6. Definition of success

The classifier is "good enough" to be useful in a real community tool if, on the held-out test set:

- **Macro-F1 ≥ 0.70**, and
- **no single class F1 < 0.60** (the model must learn *all three* boundaries, not coast on two), and
- it **beats the zero-shot baseline by ≥ 10 points** of accuracy (otherwise fine-tuning added nothing over a general model with no training).

These are objective pass/fail criteria I can check directly against the evaluation output. The per-class floor is the strict one: with ~10 test examples per class a single miss moves F1 ~0.14, so clearing 0.60 on every class means the model is genuinely separating them, not getting lucky on the easy ones.

---

## AI Tool Plan

**Label stress-testing (done up front).** Before committing to annotate 200 posts, I generated 10 deliberately boundary-sitting F1 posts and classified each by hand against the definitions above. The set surfaced the technical-vocabulary trap (§3) as the load-bearing edge case and confirmed the three labels hold under pressure. These 10 posts are recorded as the worked edge cases. If any had been genuinely unclassifiable, the definitions would have needed tightening *before* annotation — they didn't.

**Annotation assistance (disclosed if used).** I may use an LLM to pre-label a batch before reviewing, but every pre-assigned label gets read and corrected by hand — skimming pre-labels produces noisy training data and defeats the purpose. If I use this, the pre-labeled batch is tracked and disclosed in the README's AI usage section.

**Failure-pattern analysis (planned).** After evaluation, I'll paste the misclassified examples into an LLM and ask it to surface patterns (a specific label pair, post length, sarcasm, low-information posts), then re-read the examples myself to verify each pattern before it goes in the report. The model's suggestions are a starting point, not the analysis — I keep what I can confirm and note what I had to discard.

---

## Baseline note

The spec's baseline is Groq's `llama-3.3-70b-versatile`. Groq signup has errored across all three projects, so — as in Projects 1 and 2 — the zero-shot baseline runs on **Cerebras `llama-3.3-70b`** via its OpenAI-compatible endpoint (`https://api.cerebras.ai/v1`), the same underlying Meta Llama 3.3 70B model. Called with `temperature=0` and a small `max_tokens` so the model emits only the label name for clean parsing. This substitution is disclosed in the README.

---

## Stretch features (committed before starting)

Pursuing three of the four stretch features. Inter-annotator reliability is **not** being attempted — it requires a second independent human annotator I don't have lined up, and an LLM stand-in wouldn't satisfy the "two people" requirement.

### Error pattern analysis

**What:** Go past listing individual wrong predictions and name one *systematic*, generalizable error pattern with evidence from the error set.

**Plan:** My hypothesis is already on record (§3): the model over-predicts `analysis` on posts that carry technical vocabulary but are structurally bare assertions — the technical-vocabulary trap. After evaluation I'll pull every misclassified test example, count how many fit this pattern vs. other patterns, and confirm the direction in the confusion matrix (expecting a hot spot at true=`hot_take`, pred=`analysis`). I'll use an LLM to surface candidate patterns first, then verify each by re-reading the posts myself before reporting.

**Done when:** the README names one pattern, states what fraction of errors fit it, and shows at least 2–3 specific error posts as evidence — not a generic "needs more data" observation.

### Confidence calibration

**What:** Report whether the model's confidence scores are meaningful — do higher-confidence predictions actually get it right more often?

**Plan:** For each test prediction I already record the softmax max probability (needed anyway for the required sample-classifications table). I'll bin predictions by confidence and compute accuracy per bin to check whether confidence tracks correctness. With only ~30 test examples the bins are small, so I'll report this as **directional, not definitive**, and say so explicitly rather than overclaiming. A simple reliability table (or diagram) is the deliverable.

**Done when:** the README shows accuracy broken down by confidence level and states plainly whether confidence is meaningful here, with the small-sample caveat noted.

### Deployed interface

**What:** A working interface that accepts a new post, runs it through the fine-tuned classifier, and displays the predicted label and confidence.

**Plan:** A Gradio interface (`gr.Interface`) added as a cell in the Colab notebook after training, so the model is already in memory — a text box in, predicted label + confidence out (optionally the full per-class probabilities). `share=True` gives a temporary live link for the demo; the cell source is committed to the repo and run instructions go in the README. This reuses the Gradio pattern from Project 2 (FitFindr).

**Done when:** the interface accepts an arbitrary post and returns label + confidence, the code is in the repo, and the README documents how to run it.
