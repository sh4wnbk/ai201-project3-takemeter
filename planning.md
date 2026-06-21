# TakeMeter — Planning

A fine-tuned classifier that sorts Hacker News comments by what *kind* of discourse they are: a reasoned argument, a bare opinion, or a gut response. The point isn't to judge whether a take is *right* — it's to separate comments that do analytical work from comments that just assert or react.

---

## 1. Community

**Hacker News.** HN comment threads are a clean fit because a single technical thread produces all three kinds of discourse at once: someone walking through a mechanism with benchmarks, someone declaring a tool dead with no support, and someone posting a one-line quip. The community itself polices this distinction — "source?" and "[citation needed]" are native, and low-effort hot takes get challenged in replies. So the boundary I'm modeling is one regulars already recognize, not one I'm imposing from outside.

It also matches domains I know well — AI/agentic engineering, programming, quantum, space — which matters more than it sounds: the failure mode that kills these projects is *inconsistent annotation*, and I can label this discourse consistently because I can tell whether a technical claim shows a real mechanism or is decorative. Collection is weighted toward those topics for exactly that reason.

---

## 2. Labels

Three labels on a single axis: descending analytical substance.

**`analysis`** — A comment that makes a structured claim backed by specific, verifiable evidence (a mechanism, benchmarks, technical reasoning, or a citation). If you strip the opinion framing, the evidence still stands on its own as an argument.
- *"The latency win isn't the Rust rewrite — they moved the hot path off the GC'd heap; you'd get the same gain in Go with a sync.Pool, the language is incidental."*
- *"They're close to surface-code thresholds, but 'close' means you're on the pessimistic side of physical-qubits-per-logical-qubit, so the overhead is still enormous."*

**`hot_take`** — A bold, confident opinion asserted *without* real evidence. It may name-drop a technical concept, but for effect rather than as part of an argument. The claim might even be true — the comment just doesn't do the work to show it.
- *"Kubernetes is wildly over-engineered for 99% of companies. Just use a VM."*
- *"Microservices were always a mistake, full stop."*

**`reaction`** — A non-analytical comment driven by emotion or humor: a quip, a joke, snark, or short praise. There's no claim being argued — the comment is expressing a feeling or making a joke.
- *"This is the most beautiful codebase I've seen all year, wow."*
- *"lol the bot reviewing its own PR before stalebot closes it"*

These work because each boundary can be stated in one sentence, two readers applying the definitions would agree on most posts, and the distinctions reflect how the community actually talks.

---

## 3. Hard edge cases

**Primary edge case — `hot_take` vs. `analysis` (the technical-vocabulary trap).** The dangerous comment is one that *sounds* analytical but is structurally a bare assertion: *"Their architecture fundamentally can't scale — the event loop blocks on I/O, full stop."* It uses technical vocabulary ("event loop," "blocks on I/O") but offers no mechanism, no benchmark, no reasoning — just a confident verdict.

> **Decision rule:** Technical vocabulary alone does **not** make a post `analysis`. Ask whether the post supplies an actual mechanism or verifiable evidence. If removing the opinion framing leaves something that genuinely supports the claim → `analysis`. If the "evidence" is decorative — one cherry-picked stat or a technical noun with no argument behind it → `hot_take`.

This is also my predicted **model failure mode**: DistilBERT keys heavily on token-level vocabulary, so it will likely over-predict `analysis` on technical words alone. I expect the confusion matrix to show `analysis` ← `hot_take` errors concentrated on technical-sounding assertions, and I'm annotating deliberately so that prediction is testable rather than self-fulfilling.

**Secondary edge case — sarcasm between `hot_take` and `reaction`.** A deadpan sarcastic line ("oh yeah, rewriting it in Rust will *definitely* fix the architecture") carries an implied opinion but reads as a joke.

> **Decision rule:** If the post's primary act is asserting a position (even sarcastically), label `hot_take`. If its primary act is the joke/bit itself with no real position underneath, label `reaction`. Ask: is the author arguing, or performing?

---

## 4. Data collection plan

**Source:** Public Hacker News comments via the Algolia HN Search API (`hn.algolia.com/api/v1/search?tags=comment`) — open, no auth, no keys. Collection is weighted across six debate-heavy technical topics I know well: `llm agents`, `rust borrow checker`, `kubernetes`, `quantum error correction`, `startup failure`, `rocket engine`.

**Target:** ~250–270 candidates, **even quota per query** (~45 each) so no single topic dominates — collecting proportional to availability would let one domain, and its dominant label, swamp the set and let the model learn *topic* instead of *discourse type*. No single label may exceed 70%.

**The expected imbalance, and the fix:** Comment threads skew toward `reaction` and `hot_take`; genuine `analysis` is rarer and clusters in technical debate. I bias the queries toward debate-heavy threads to surface it — validated at ~25–35% `analysis` in those threads — and dedup on `objectID` across queries so the same comment can't leak across the train/test split. Boilerplate ("Who is hiring" threads) is filtered by `story_title`. If `analysis` is still thin after a pass, I pull a second targeted batch rather than padding the other classes, which would just train a majority-class predictor.

**Annotation:** Comments are pre-labeled by Cerebras `llama-3.3-70b` as drafts (`label_suggested`), then **every row is reviewed and corrected by hand**, with final labels written to a separate `label` column so the draft history survives. Disclosed in the README's AI usage section.

**Format:** Final CSV has columns `text`, `label`, `notes`. The notebook handles the 70/15/15 split — one labeled file, not pre-split.

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

These are objective pass/fail criteria I can check directly against the evaluation output. The per-class floor is the strict one: with ~250 labeled examples a 15% test split leaves ~12–13 examples per class, where a single miss still moves F1 by ~0.08–0.12, so clearing 0.60 on every class means the model is genuinely separating them, not getting lucky on the easy ones.

---

## AI Tool Plan

**Label stress-testing (done up front).** Before annotating, I stress-tested the labels with ~10 deliberately boundary-sitting comments — the technical-vocabulary trap (jargon with no mechanism), sarcasm sitting between `hot_take` and `reaction`, and argued-but-trailing-off comments — and classified each by hand against the definitions above. The set surfaced the technical-vocabulary trap (§3) as the load-bearing edge case and confirmed the three labels hold under pressure. If any had been genuinely unclassifiable, the definitions would have needed tightening *before* annotation — they didn't.

**Annotation assistance (used and disclosed).** I use Cerebras `llama-3.3-70b` to pre-label the collected comments as drafts, then read and correct every row by hand — skimming pre-labels produces noisy training data and defeats the purpose. The pre-labeled batch is tracked (a `prelabeled` flag, draft kept in a separate `label_suggested` column) and disclosed in the README's AI usage section, including how many drafts I overrode.

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
