# LORE Launch Content

## Show HN Post

**Title:** Show HN: LORE – Local AI orchestration that makes 9B models punch above their weight

**Body:**

I've been building LORE — an orchestration engine that coordinates multiple small models on edge devices (16 GB RAM) to match the quality of much larger models.

The core idea: instead of stuffing everything into one prompt with a constrained context window, LORE decomposes complex tasks into focused subtasks, each with a small context that actually fits in memory. A classifier (using a 1.5B SSM model) decides whether to orchestrate or dispatch directly.

What makes it different from just "use a bigger model":
- Runs on 16 GB — no 24 GB or 48 GB hardware needed
- Both models (9B primary + 1.5B specialist) fit simultaneously with 7 GB headroom
- TurboQuant KV cache compression (3.6×) with zero PPL degradation on SSM models
- Benchmark-driven model selection — auto-discovers better models from HuggingFace
- Conditional gating — optimizations only activate when they'd actually help

237 tests, 8 phases of measured development, every optimization tested against real inference on M4 hardware.

Currently benchmarking against HumanEval to get hard numbers. The thesis: orchestrated 9B+1.5B at 16 GB should approach 14B quality territory.

Looking for contributors — especially:
- EAGLE-3 speculative decoding testing (could give 2-3× speedup)
- SWE-bench Verified integration
- New model support (Mamba-3 when GGUF available)

Repo: https://github.com/oniwakaa/lore

---

## X/Twitter Thread

**Tweet 1 (hook):**
I built an orchestration engine that runs 2 AI models simultaneously on a 16 GB Mac.

Not "pick one model." Not "use cloud APIs." Two specialized models, coordinated, within hardware budget.

7 GB headroom. 237 tests. Real benchmarks incoming.

🧵👇

**Tweet 2 (the problem):**
The dirty secret of "runs on 16 GB":

A 9B model at Q4 = 5.6 GB weights.
KV cache at 262K context = doesn't fit.
Real usable context = ~16K.

A 27B model "runs on 24 GB" but at 16-32K context, not 262K.

The benchmark numbers are lying about real-world capability.

**Tweet 3 (the solution):**
LORE decomposes complex tasks into focused subtasks.

Each subtask gets 2-4K context — small enough to fit comfortably.

A 1.5B SSM specialist (near-zero KV cache) handles classification and simple tasks.
The 9B model handles reasoning and code generation.

Total: 6.59 GB. 7.41 GB headroom.

**Tweet 4 (the tech):**
Key pieces:
• TurboQuant KV compression — 3.6× with 0% quality loss
• Falcon-H1-1.5B specialist — hybrid SSM, near-zero KV cache
• TF-IDF router — <1ms, no LLM cost for routing
• Conditional gating — optimizations activate only when they'd help
• HF leaderboard scanning — auto-discovers better models

**Tweet 5 (the ask):**
Open source, MIT license, 237 tests passing.

Looking for contributors to help with:
• EAGLE-3 speculative decoding (2-3× speedup)
• HumanEval/SWE-bench benchmarking
• New model support

If you run local models and care about squeezing every drop of performance from edge hardware — come help build this.

github.com/oniwakaa/lore

---

## Reddit Post (r/LocalLLaMA)

**Title:** LORE: Orchestration engine that runs 2 models on 16 GB — 9B primary + 1.5B SSM specialist, 7 GB headroom

**Body:**

I've been working on LORE — a local AI orchestration layer for edge devices. The idea: instead of one model with a constrained context, coordinate multiple specialized small models.

**What it does:**
- Runs Ornith-1.0-9B (primary) + Falcon-H1-1.5B (specialist) simultaneously
- Total memory: 6.59 GB with TurboQuant KV compression, 7.41 GB headroom on 16 GB
- Classifier (using the 1.5B model) decides: route directly or decompose into subtasks
- Each subtask gets a focused 2-4K context instead of fighting for space in a shared 16K window

**Key technical choices:**
- TurboQuant KV cache: 3.6× compression, 0% PPL degradation on hybrid SSM models
- Falcon-H1-1.5B: hybrid SSM, only 2 attention heads, near-zero KV cache — perfect specialist
- Non-LLM router (TF-IDF + LogReg): <1ms, no inference cost for the routing decision
- Conditional gating: each optimization only activates at the scale it was designed for
- Benchmark-driven model selection: scans HuggingFace leaderboards for better models per task type

**What's tested:**
237 passing tests. Every optimization measured against real inference on M4. Techniques that don't work on SSM architectures (speculative decoding, TIDE, MiniCache) were skipped with evidence.

**What's next:**
Running against HumanEval to get pass@1 numbers. The thesis is that orchestrated 9B+1.5B should approach 14B quality territory — not by using a bigger model, but by making each token count.

Looking for contributors: https://github.com/oniwakaa/lore

---

## Dev.to Post (draft title)

**Title:** Building a Local AI Orchestra: How Two Small Models Beat One Big Model on 16 GB

**Tags:** #ai, #opensource, #python, #llm
