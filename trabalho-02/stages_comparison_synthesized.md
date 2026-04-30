# Stages Comparison — Pergunta / Resposta / Modelo usado

3 grafos sobre o mesmo corpus, mesma query em cada um.

**Modelo usado em todas as queries:** claude-sonnet (via `claude -p --model sonnet`) com a **skill graphify** (BFS subgraph traversal sobre `graphify-out/graph.json`)

**Stages:**
- **Stage 1** — papers só (54 arquivos · 118 nodes · 168 edges)
- **Stage 2** — papers + docs lib peft/trl/bnb (131 arquivos · 294 nodes · 368 edges)
- **Stage 3** — papers + docs lib + 1058 .py (1189 arquivos · 18.803 nodes · 66.935 edges)

---

## Q01

**Pergunta:** When to use LoRA vs QLoRA vs full fine-tuning for low-GPU devices

**Modelo usado:** claude-sonnet (via `claude -p --model sonnet`) com a **skill graphify** (BFS subgraph traversal sobre `graphify-out/graph.json`)

### Stage 1 (papers só (54 arquivos · 118 nodes · 168 edges))

**Resposta:**

## LoRA vs QLoRA vs Full Fine-tuning for Low-GPU Devices

### Decision rule

| Scenario | Recommended |
|---|---|
| VRAM < 12 GB (single consumer GPU) | **QLoRA** |
| VRAM 12–24 GB, moderate domain shift | **LoRA** |
| VRAM 40 GB+, maximum domain learning needed | **Full fine-tuning** |

---

### Full Fine-tuning — avoid on low-GPU

A 7B model requires ~28 GB in BF16, scaling to ~260 GB for 65B. Beyond raw VRAM, Biderman et al. (2024) show full FT uses weight updates with **10–100× higher rank** than LoRA, making it the only option when *maximum target-domain performance is the hard requirement*. But it **forgets more** — general capabilities erode faster than with LoRA.

*Source: `02-lora/08-lora-learns-less-forgets-less.md` — arxiv:2405.09673*

---

### LoRA — good middle ground

Freezes the base model; trains small rank-decomposition matrices (ΔW = BA, r ≪ d). Reduces trainable parameters by **10,000× vs GPT-3 full FT** and GPU memory by **3×**. No inference latency since adapters merge back. Key tradeoff: **learns less** in the target domain vs full FT (rank mismatch), but **forgets less** of the base model's general capabilities. If you need to preserve generality, LoRA wins over full FT regardless of GPU constraints.

*Source: `02-lora/01-lora-original.md` — arxiv:2106.09685; `02-lora/08-lora-learns-less-forgets-less.md` — arxiv:2405.09673*

---

### QLoRA — the low-GPU workhorse

QLoRA = 4-bit quantized frozen base (NF4) + LoRA adapters. Three stacked innovations drive the memory savings:

- **NF4**: information-theoretically optimal for normally distributed weights, outperforms INT4/FP4
- **Double Quantization**: quantizes the quantization constants, saving ~0.37 bits/param
- **Paged Optimizers**: spills optimizer states to CPU RAM during gradient-checkpointing spikes

Result:

| Model | Full FT (BF16) | QLoRA (NF4) |
|---|---|---|
| 7B | ~28 GB | **~6 GB** |
| 13B | ~52 GB | **~10 GB** |
| 65B | ~260 GB | **~48 GB** |

Guanaco (65B fine-tuned via QLoRA on a single 48 GB GPU in 24 h) reached **99.3% of ChatGPT performance** on the Vicuna benchmark. Quality loss vs standard LoRA is negligible for instruction tuning; data quality matters more than the quantization gap.

*Source: `02-lora/02-qlora.md` + `06-quantization/01-qlora-quantization.md` — arxiv:2305.14314 (Dettmers et al., NeurIPS 2023)*

---

### Summary

The graph's **"Quantization-Enabled Fine-Tuning Stack"** hyperedge (`01_qlora_quantization_qlora`, `01_qlora_nf4`, `01_qlora_double_quantization`, `04_bitsandbytes_lib`, `01_lora_original_lora`) captures the practical stack: **QLoRA is simply LoRA run on top of a quantized model**, so it inherits all LoRA properties (no inference overhead, adapter merging, low forgetting) while slashing VRAM by ~4–5×. For any device with < 24 GB VRAM, QLoRA is the default choice; full fine-tuning is only justified when you have the hardware and need maximum domain adaptation at the cost of forgetting.

### Stage 2 (papers + docs lib peft/trl/bnb (131 arquivos · 294 nodes · 368 edges))

**Resposta:**

## LoRA vs QLoRA vs Full Fine-Tuning on Low-GPU Devices

### Decision Framework

**Full fine-tuning** trains all model parameters, requiring BF16 activations + optimizer states across every weight. The graph's `QLoRA` node (`02-lora/02-qlora.md`) shows the raw cost: a 7B model needs ~28 GB GPU RAM, 13B needs ~52 GB, 65B needs ~260 GB. This rules out full fine-tuning entirely for consumer-grade GPUs (≤24 GB) on anything beyond tiny models.

**LoRA** (reparameterized PEFT) freezes the base model in BF16/FP16 and trains only small low-rank matrices (A, B) injected into transformer attention layers. Per `02-lora/01-lora-original.md`, this yields a ~3× reduction in GPU memory and 10,000× fewer trainable parameters for GPT-3 scale, with no inference latency penalty (adapters merge at inference). It's the right choice when you can load the base model in BF16 — roughly when the model fits on your GPU at half precision.

**QLoRA** adds 4-bit NF4 quantization of the frozen base weights on top of LoRA. Per `02-lora/02-qlora.md` and `06-quantization/01-qlora-quantization.md`, this cuts a 7B model to ~6 GB, 13B to ~10 GB, 65B to ~48 GB — enabling single-GPU fine-tuning of models that otherwise require multi-GPU or won't load at all. Three innovations make it lossless in practice: NF4 (information-theoretically optimal for normally distributed weights), Double Quantization (saves ~0.37 bits/param on constants), and Paged Optimizers (prevents OOM spikes via CPU memory paging). The `01-sft/02-peft-comprehensive-survey-2024.md` confirms QLoRA retains 90–95% of full fine-tuning quality.

### Rule of Thumb

| GPU VRAM | Model Size | Method |
|---|---|---|
| ≥ full BF16 footprint | Any | Full fine-tuning (if budget allows) |
| < BF16, > 4-bit footprint | Fits in FP16 | LoRA |
| Very limited (≤12 GB) | 7B–13B | QLoRA |
| Extreme (single 48GB) | 65B+ | QLoRA only viable path |

**QLoRA is the default for low-GPU devices**: it reaches full fine-tuning quality (Guanaco hits 99.3% of ChatGPT on Vicuna after 24h on one GPU) while keeping VRAM at ~6 GB for a 7B model. Use plain LoRA only when quantization overhead is a concern (e.g., inference serving of the adapter) or the model already fits in BF16.

**Sources:** `02-lora/02-qlora.md` (https://arxiv.org/abs/2305.14314), `06-quantization/01-qlora-quantization.md` (https://arxiv.org/abs/2305.14314), `02-lora/01-lora-original.md` (https://arxiv.org/abs/2106.09685), `01-sft/02-peft-comprehensive-survey-2024.md` (https://arxiv.org/abs/2403.14608)

### Stage 3 (papers + docs lib + 1058 .py (1189 arquivos · 18.803 nodes · 66.935 edges))

**Resposta:**

## LoRA vs QLoRA vs Full Fine-Tuning: Decision Guide for Low-GPU Devices

### The Core Tradeoff

The graph's **Quantization-Enabled Fine-Tuning Stack** hyperedge (confidence 0.95) links `QLoRA → NF4 → Double Quantization → bitsandbytes → LoRA` as a coherent technology chain, not alternatives in isolation.

---

### Full Fine-Tuning — Avoid on Low-GPU Devices

The graph's OSF continual-learning node (`code\peft\examples\orthogonal_subspace_learning\osf_continual_learning.py:L395`) treats full fine-tuning purely as a "baseline for comparison." The LoRA paper (`02-lora/01-lora-original.md`, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) quantifies why: full fine-tuning a 7B model at BF16 needs **~28 GB** of VRAM (weights + optimizer states + gradients). Only viable on multi-GPU or A100-class hardware.

---

### LoRA (Plain) — 16–24 GB VRAM

LoRA freezes the base model and trains only low-rank adapter matrices (B·A), reducing trainable parameters by **~10,000×** and GPU memory by **~3×** vs. full fine-tuning (LoRA paper). A 7B model in FP16 still requires ~14 GB for the frozen base alone — feasible on an RTX 3090/4090 or A6000. Advantages: no quantization error, fastest iteration, adapters merge at inference with zero latency.

**Use when:** 16–24 GB VRAM available, quality is priority.

---

### QLoRA — 6–12 GB VRAM (the key low-GPU choice)

QLoRA (`06-quantization/01-qlora-quantization.md`, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)) adds three innovations on top of LoRA:

| Component | Effect |
|---|---|
| **NF4 quantization** of frozen base | ~25% of FP16 memory |
| **Double Quantization** (8-bit quant constants) | saves additional 0.37 bits/param |
| **Paged Optimizers** | handles gradient-checkpoint memory spikes |

Result per the QLoRA paper and bitsandbytes docs (`06-quantization/04-bitsandbytes-docs.md`, [HF docs](https://huggingface.co/docs/bitsandbytes/main/en/index)):

| Model | BF16 Training | QLoRA (NF4+DQ) |
|---|---|---|
| 7B | ~28 GB | **~6 GB** |
| 13B | ~52 GB | **~10 GB** |
| 33B | ~132 GB | **~24 GB** |

The paper claims task performance **matches full fine-tuning** despite quantization. Trade-off: minor weight quantization error (NF4 is information-theoretically optimal for Gaussian weights, so error is minimal), and slower forward passes due to dequantize/quantize overhead.

**Use when:** VRAM < 16 GB; especially 8–12 GB consumer GPUs (RTX 3080/4070/4080).

---

### Decision Rule

```
VRAM < 8 GB   → QLoRA, ≤7B model, load_in_4bit + nf4 + double_quant
8–16 GB       → QLoRA, 7B–13B model
16–24 GB      → Plain LoRA on FP16 base (7B), higher quality
24 GB+        → Full fine-tuning (small models) or LoRA (large models)
```

**Configuration reference** (`06-quantization/04-bitsandbytes-docs.md`):
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # saves ~0.37 bits/param
)
```

**Sources:** `06-quantization/01-qlora-quantization.md` ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)), `06-quantization/04-bitsandbytes-docs.md` ([HF bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/main/en/index)), `02-lora/01-lora-original.md` ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)), `02-lora/07-hf-peft-lora-docs.md` ([HF PEFT docs](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)), `code\peft\examples\orthogonal_subspace_learning\osf_continual_learning.py:L395`.

---

## Q02

**Pergunta:** VRAM tradeoff between LoRA, QLoRA and Full Fine-Tuning on a 7B and 13B model

**Modelo usado:** claude-sonnet (via `claude -p --model sonnet`) com a **skill graphify** (BFS subgraph traversal sobre `graphify-out/graph.json`)

### Stage 1 (papers só (54 arquivos · 118 nodes · 168 edges))

**Resposta:**

## VRAM Tradeoff: Full Fine-Tuning vs LoRA vs QLoRA

### Concrete numbers (7B and 13B)

| Method | 7B | 13B |
|---|---|---|
| **Full Fine-Tuning** (BF16) | ~28 GB | ~52 GB |
| **LoRA** (16-bit frozen base) | ~9 GB* | ~17 GB* |
| **QLoRA** (4-bit NF4) | ~6 GB | ~10 GB |

*Derived from the LoRA paper's stated 3× memory reduction over full FT.

Sources: `02-lora/02-qlora.md` and `06-quantization/01-qlora-quantization.md` (both citing arXiv:2305.14314); `02-lora/01-lora-original.md` (arXiv:2106.09685).

---

### Why each method lands where it does

**Full Fine-Tuning** is the most expensive because every parameter requires weights + gradients + Adam optimizer states (m and v) in mixed precision — roughly 12–16 bytes/parameter. A 7B model hits ~28 GB before any activation memory.

**LoRA** (`02-lora/01-lora-original.md`) freezes the backbone: the base weights stay at 2 bytes/param (BF16) with no gradients or optimizer states. Only the tiny rank-decomposition adapters (typically <1% of parameters, covering attention projections Wq/Wv) need gradient tracking. The LoRA paper reports a **3× reduction** in GPU memory versus full FT.

**QLoRA** (`06-quantization/01-qlora-quantization.md`) stacks two further compressions on top of LoRA:
1. **NF4 quantization** — base weights stored at 4 bits (~0.5 bytes/param), information-theoretically optimal for normally distributed weights.
2. **Double Quantization** — quantizes the 32-bit quantization constants to 8-bit, saving an additional ~0.37 bits/parameter.
3. **Paged Optimizers** — spills adapter optimizer states to CPU RAM during gradient-checkpointing spikes, preventing OOM errors.

The result: a 65B model fits on a single 48 GB GPU; a 7B model fits on an 8 GB consumer GPU, enabling single-GPU fine-tuning at consumer scale (`02-lora/02-qlora.md`).

---

**Practical rule of thumb from the subgraph:** QLoRA cuts VRAM by ~4.5–5× versus full FT; plain LoRA by ~3×. Quality is preserved — QLoRA matches full 16-bit FT performance (arXiv:2305.14314), while LoRA matches or exceeds full FT on standard benchmarks (arXiv:2106.09685).

### Stage 2 (papers + docs lib peft/trl/bnb (131 arquivos · 294 nodes · 368 edges))

**Resposta:**

## VRAM Tradeoffs: Full Fine-Tuning vs. LoRA vs. QLoRA (7B & 13B)

### The Numbers

| Model | Full FT (BF16) | LoRA (BF16 base) | QLoRA (NF4 4-bit) |
|-------|---------------|------------------|--------------------|
| 7B    | ~28 GB        | ~14–16 GB        | ~6 GB              |
| 13B   | ~52 GB        | ~26–28 GB        | ~10 GB             |

*Full FT and QLoRA figures are directly from the QLoRA paper table. LoRA (BF16) is derived from: base weights frozen in 16-bit (~2 bytes/param) + tiny adapter overhead.*

### Why the Gap Exists

**Full Fine-Tuning** stores: 16-bit weights + 16-bit gradients for every parameter + Adam optimizer states (2× model size). For a 7B model this compounds to ~28 GB minimum, with activations on top.

**LoRA** (`02-lora/01-lora-original.md`, arXiv:2106.09685) freezes the base weights—eliminating gradient storage for the bulk of parameters. Only the low-rank adapter matrices A and B (rank r ≪ d) carry gradients and optimizer states. The base model stays in BF16, so VRAM scales roughly linearly with model size. The original paper claims ~3× reduction versus full FT at GPT-3 scale.

**QLoRA** (`02-lora/02-qlora.md`, arXiv:2305.14314) compounds three innovations on top of LoRA: **(1) NF4 4-bit quantization** of base weights (information-theoretically optimal for normally distributed weights), **(2) Double Quantization** — quantizing the quantization constants themselves, saving ~0.37 bits/parameter — and **(3) Paged Optimizers** to handle memory spikes via CPU offload. Together these reduce a frozen 7B base from ~14 GB to ~5.5 GB, yielding ~6 GB total including adapters.

### Key Tradeoff

QLoRA's ~4–5× VRAM advantage over LoRA comes at the cost of dequantization overhead during the forward/backward pass (NF4 → BF16 compute). The QLoRA paper reports this preserves full 16-bit fine-tuning task performance, but throughput is lower than BF16 LoRA on the same hardware.

**Sources:** `02-lora/02-qlora.md` (arXiv:2305.14314), `06-quantization/01-qlora-quantization.md` (arXiv:2305.14314), `02-lora/01-lora-original.md` (arXiv:2106.09685)

### Stage 3 (papers + docs lib + 1058 .py (1189 arquivos · 18.803 nodes · 66.935 edges))

**Resposta:**

## VRAM Tradeoff: LoRA vs QLoRA vs Full Fine-Tuning (7B & 13B)

### Memory Requirements

| Model | Full FT (BF16) | LoRA (16-bit) | QLoRA (NF4 4-bit) |
|-------|---------------|---------------|-------------------|
| 7B    | ~28 GB        | ~9 GB (3x reduction) | ~6 GB |
| 13B   | ~52 GB        | ~17 GB (3x reduction) | ~10 GB |

### Full Fine-Tuning
Requires loading all parameters in BF16/FP16 plus optimizer states (Adam doubles memory). At 16-bit, a 7B model needs ~28 GB and a 13B needs ~52 GB — exceeding a single consumer or even prosumer GPU. **Source:** `02-lora/02-qlora.md`, `06-quantization/01-qlora-quantization.md`

### LoRA
Freezes base weights and trains only low-rank adapter matrices injected into attention layers. Reduces GPU memory requirements by ~3x versus full fine-tuning, and reduces trainable parameters by ~10,000x at GPT-3 scale. Checkpoint size is ~10,000x smaller. Training cost is ~25% of Adam full fine-tuning. **Source:** `02-lora/01-lora-original.md` (https://arxiv.org/abs/2106.09685)

### QLoRA
Stacks quantization on top of LoRA: base weights are frozen in NF4 (4-bit), adapters train in BF16. Three mechanisms drive the savings:
- **NF4** quantization of frozen weights (4 bits vs 16)
- **Double Quantization** of the quantization constants themselves (saves an additional ~0.37 bits/param, reducing effective width from 4.5 to 4.37 bits/param)
- **Paged optimizers** to handle gradient checkpointing spikes

Result: a 7B model fits in ~6 GB; a 13B in ~10 GB — roughly **78–81% reduction** vs full fine-tuning, and ~33% below standard LoRA. Critically, QLoRA matches full 16-bit fine-tuning task performance. **Source:** `02-lora/02-qlora.md`, `06-quantization/01-qlora-quantization.md` (https://arxiv.org/abs/2305.14314)

### Bottom Line
For 7B/13B models on a single GPU: **Full FT is infeasible without multi-GPU**, **LoRA makes it possible**, **QLoRA makes it cheap** — fitting a 13B fine-tune in a single 12–16 GB VRAM card. The LoRA family offers the best accuracy-vs-parameter-efficiency tradeoff for most LLMs. **Source:** `03-adapters-prompts/05-peft-survey-methodologies-2024.md`

---

## Q03

**Pergunta:** Quality degradation tradeoff: LoRA vs full fine-tuning on standard benchmarks

**Modelo usado:** claude-sonnet (via `claude -p --model sonnet`) com a **skill graphify** (BFS subgraph traversal sobre `graphify-out/graph.json`)

### Stage 1 (papers só (54 arquivos · 118 nodes · 168 edges))

**Resposta:**

## Quality Degradation Tradeoff: LoRA vs Full Fine-Tuning

### Core Finding: LoRA Learns Less, Forgets Less

The most direct evidence comes from Biderman et al. (2024), **"LoRA Learns Less and Forgets Less"** (`02-lora/08-lora-learns-less-forgets-less.md`, https://arxiv.org/abs/2405.09673). Across programming and mathematics domains — tested in both instruction fine-tuning (~100K pairs) and continued pretraining (20B tokens) — standard LoRA configurations **underperform full fine-tuning on target-domain benchmarks**. The mechanism is clear: full fine-tuning applies weight perturbations with **10–100× higher effective rank** than typical LoRA (r=8 or r=16). This rank mismatch is the primary driver of the quality gap. Importantly, high-rank LoRA (r=256+) substantially closes this gap.

The tradeoff is symmetric: LoRA **better preserves general capabilities** outside the target domain, outperforming dropout and weight decay as regularizers against catastrophic forgetting.

### Confirming the Gap

DoRA's paper (Liu et al., 2024, `02-lora/03-dora.md`, https://arxiv.org/abs/2402.09353) independently confirms: *"there is still an accuracy gap between LoRA-based methods and full fine-tuning."* DoRA was designed precisely to close this gap by decomposing weights into magnitude and direction, learning them separately — and it outperforms standard LoRA on commonsense reasoning benchmarks.

The PEFT comprehensive survey (Han et al., 2024, `01-sft/02-peft-comprehensive-survey-2024.md`, https://arxiv.org/abs/2403.14608) quantifies the general baseline: PEFT methods retain **90–95% of full fine-tuning quality** while reducing trainable parameters by 10–1000×.

### Summary

| | Target-domain quality | General capabilities |
|---|---|---|
| Full fine-tuning | Higher | Degrades more |
| Standard LoRA (r=8–16) | ~90–95% | Better preserved |
| High-rank LoRA (r=256+) | Approaches full FT | Near full FT degradation |

The rank of the LoRA adapter is the primary control knob for this tradeoff.

### Stage 2 (papers + docs lib peft/trl/bnb (131 arquivos · 294 nodes · 368 edges))

**Resposta:**

## LoRA vs Full Fine-Tuning: Quality Degradation Tradeoff

### Core claim from the original paper

The LoRA paper (`02-lora/01-lora-original.md`, arXiv:2106.09685) claims LoRA achieves **comparable or superior performance** to full fine-tuning on RoBERTa, DeBERTa, GPT-2, and GPT-3 using rank r=4. On GPT-3 175B specifically, LoRA matched full fine-tuning at ~25% of training cost and 10,000× smaller checkpoints. The PEFT documentation (`code/peft/docs/source/conceptual_guides/adapter.md`) echoes this: "Performance of models finetuned using LoRA is comparable to the performance of fully finetuned models."

### The gap is real — and acknowledged

Despite these claims, DoRA (`02-lora/03-dora.md`, arXiv:2402.09353) explicitly opens with: *"there is still an accuracy gap between LoRA-based methods and full fine-tuning."* DoRA's existence as a paper is itself evidence that the gap is real in practice — particularly on commonsense reasoning and vision-language benchmarks. DoRA closes part of this gap by decomposing updates into magnitude and direction, showing that standard LoRA constrains expressivity in ways FT does not.

### QLoRA adds a further layer

QLoRA (`06-quantization/01-qlora-quantization.md`, arXiv:2305.14314) claims to **recover full 16-bit FT performance** while running in 4-bit NF4 quantization — meaning quantization itself is not the primary quality cost when NF4 + double quantization are used correctly. The quality gap, when it exists, originates from LoRA's low-rank constraint, not quantization.

### Summary

| Factor | Effect on quality gap |
|---|---|
| LoRA at r=4 | Matches FT on large models (GPT-3, RoBERTa) per original paper |
| LoRA in practice | Gap persists on reasoning/multimodal tasks (DoRA motivation) |
| QLoRA (4-bit) | No additional gap vs LoRA — NF4 quantization is lossless at task level |
| DoRA | Narrows the gap further by separating magnitude/direction updates |

**Bottom line:** The quality gap is rank-constrained and task-dependent — larger, more expressive models close it more easily; reasoning-heavy tasks expose it more.

### Stage 3 (papers + docs lib + 1058 .py (1189 arquivos · 18.803 nodes · 66.935 edges))

**Resposta:**

## LoRA vs Full Fine-Tuning: Quality Degradation Tradeoff

**Core finding: standard LoRA underperforms full fine-tuning on domain-specific benchmarks, with the gap explained by rank mismatch.**

### The Quality Gap

The original LoRA paper (2106.09685) claimed "comparable or superior performance" to full fine-tuning on RoBERTa, DeBERTa, and GPT-3 benchmarks at r=4, while reducing trainable parameters by 10,000× and GPU memory by 3×. This was the optimistic early view.

Later research revised this. The "LoRA Learns Less, Forgets Less" paper (2405.09673) identifies the mechanism: **full fine-tuning uses perturbations with 10–100× higher effective rank** than typical LoRA (r=8 or r=16). This rank deficit is the primary explanatory factor for the performance gap in domain adaptation tasks. The tradeoff is directional:

- **Full fine-tuning**: learns more in-domain, forgets more out-of-domain
- **Standard LoRA**: learns less in-domain (quality gap), forgets less out-of-domain (better generalization preservation)

High-rank LoRA (r=256+) can close the quality gap but erodes the parameter-efficiency advantage.

### Variants That Narrow the Gap

The graph's "LoRA Variant Family" hyperedge clusters the responses: **DoRA** (2402.09353) decomposes weights into magnitude + direction, demonstrating improvements over standard LoRA on commonsense reasoning (LLaMA) and visual instruction tuning, explicitly acknowledging "an accuracy gap between LoRA-based methods and full fine-tuning." **AdaLoRA** (2303.10512) addresses it via adaptive SVD-based rank allocation. **QLoRA** (2305.14314) claims to preserve "full 16-bit finetuning task performance" at 4-bit precision, reaching 99.3% of ChatGPT on Vicuna.

### Bottom Line

The 2024 PEFT survey (2410.19878) concludes the LoRA family offers the best accuracy-vs-parameter-efficiency tradeoff for most LLMs — but "best tradeoff" is not parity. A measurable quality gap on domain benchmarks persists for standard low-rank configs; closing it requires either higher rank, a DoRA/AdaLoRA variant, or accepting the efficiency cost.

**Sources:** `02-lora/01-lora-original.md`, `02-lora/08-lora-learns-less-forgets-less.md`, `02-lora/03-dora.md`, `02-lora/02-qlora.md`, `02-lora/05-adalora.md`, `03-adapters-prompts/05-peft-survey-methodologies-2024.md`

---

## Q04

**Pergunta:** Which fine-tuning method works on a single RTX 4090 24GB?

**Modelo usado:** claude-sonnet (via `claude -p --model sonnet`) com a **skill graphify** (BFS subgraph traversal sobre `graphify-out/graph.json`)

### Stage 1 (papers só (54 arquivos · 118 nodes · 168 edges))

**Resposta:**

## QLoRA is the method designed for a single RTX 4090 (24 GB)

The graph's central answer comes from two nodes in community 8 — **QLoRA** and **QLoRA Quantization** — both sourced from the same paper (Dettmers et al., 2023, NeurIPS).

### What the evidence shows

The memory table extracted from `06-quantization/01-qlora-quantization.md` ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)) is definitive:

| Model Size | Full FT (BF16) | **QLoRA (NF4 4-bit)** |
|------------|---------------|----------------------|
| 7B         | ~28 GB        | **~6 GB**            |
| 13B        | ~52 GB        | **~10 GB**           |
| 33B        | ~132 GB       | **~24 GB** ← fits exactly |
| 65B        | ~260 GB       | ~48 GB (needs 2×GPU) |

A 7B or 13B model fits easily; a 33B model fits right at the 24 GB ceiling.

### The enabling technique stack

The graph traces the path: `QLoRA` --implements--> **NF4 (4-bit NormalFloat)** + **Double Quantization** --references--> **bitsandbytes Library**. These three innovations together achieve the memory reduction:
- NF4 stores weights in 4-bit (information-theoretically optimal for normally-distributed weights)
- Double Quantization saves an additional ~0.37 bits/param by quantizing the quantization constants
- Paged Optimizers handle memory spikes via CPU offload

### Runner-up: EfficientQAT (partial fit)

`06-quantization/02-efficientqat.md` ([arXiv:2407.11062](https://arxiv.org/abs/2407.11062)) shows its Phase 1 (block-wise training) requires ~24 GB for a 7B model — technically on the edge — but Phase 2 requires ~40 GB, so it does **not** fully run on a single 4090.

**Bottom line:** QLoRA (via NF4 + bitsandbytes) is the graph's clear answer for single RTX 4090 fine-tuning, supporting models up to 33B parameters within the 24 GB limit.

### Stage 2 (papers + docs lib peft/trl/bnb (131 arquivos · 294 nodes · 368 edges))

**Resposta:**

## QLoRA: The Answer for a Single RTX 4090 (24 GB)

**QLoRA** (Quantized Low-Rank Adaptation) is the fine-tuning method designed to run on constrained GPU memory. The evidence from the graph nodes directly answers this:

### Why QLoRA fits a 24 GB GPU

The memory table from the QLoRA paper (`06-quantization/01-qlora-quantization.md`, `02-lora/02-qlora.md`, source: https://arxiv.org/abs/2305.14314) shows:

| Model Size | BF16 Full Fine-tune | QLoRA (NF4 4-bit) |
|---|---|---|
| 7B | ~28 GB ❌ | ~6 GB ✅ |
| 13B | ~52 GB ❌ | ~10 GB ✅ |
| 33B | ~132 GB ❌ | ~24 GB ✅ (fits exactly) |
| 65B | ~260 GB ❌ | ~48 GB ❌ |

A 24 GB RTX 4090 can run QLoRA for **7B, 13B, and up to 33B models**. Full fine-tuning fails even for a 7B model (~28 GB required).

### Three innovations that make it possible

All from `06-quantization/01-qlora-quantization.md`:
1. **NF4 (4-bit NormalFloat)** — quantizes frozen base weights to 4-bit, information-theoretically optimal for normally distributed LLM weights
2. **Double Quantization** — quantizes the quantization constants too, saving ~0.37 bits/param
3. **Paged Optimizers** — spills optimizer states to CPU RAM during gradient checkpointing spikes, preventing OOM

LoRA adapters (trained in BF16) are injected into the frozen quantized model via the `bitsandbytes` library + HuggingFace PEFT (`code/peft/docs/source/task_guides/lora_based_methods.md`). The Guanaco 65B result (trained on a single 48 GB GPU in 24 hours reaching 99.3% of ChatGPT) validates the method's efficiency claim.

**TL;DR:** QLoRA is the only method in the graph that fits a 24 GB GPU for models ≥7B. Plain LoRA without quantization still requires full BF16 weights (~28 GB for 7B alone).

### Stage 3 (papers + docs lib + 1058 .py (1189 arquivos · 18.803 nodes · 66.935 edges))

**Resposta:**

## Fine-tuning on a Single RTX 4090 (24 GB): What the Graph Shows

**The primary method is QLoRA** — LoRA adapters applied on top of a 4-bit quantized base model via bitsandbytes.

### Core stack

The graph contains a direct path from bitsandbytes 4-bit quantization to PEFT adapter training:

1. **`Linear4bit` / `LinearNF4`** (bitsandbytes) — the base quantization layer. A rationale node explicitly states: *"This class is the base module for the 4-bit quantization algorithm"* → source: `code\bitsandbytes\bitsandbytes\nn\modules.py`

2. **`peft/tuners/lora/bnb.py`** — the dedicated PEFT integration file that wraps `Linear4bit` and `Linear8bitLt` with LoRA merge/unmerge logic → source: `code\peft\src\peft\tuners\lora\bnb.py`

3. **`prepare_model_for_kbit_training()`** — the TRL utility that wires it together: *"Prepare a k-bit quantized transformers model for training (PEFT/QLoRA)"* → source: `code\trl\trl\experimental\utils.py` (L401)

### Confirmed single-GPU methods (4-bit)

The graph's `PeftBnbGPUExampleTests` test class (*"A single GPU int8 + fp4 test suite, this will test if training fits correctly"* — `code\peft\tests\test_gpu_examples.py` L172) validates all of these on a single GPU:

| Method | Graph evidence node | Source file |
|---|---|---|
| **LoRA (QLoRA)** | `test_causal_lm_training_4bit` | `code\peft\tests\test_gpu_examples.py` |
| **DoRA** | `test_causal_lm_training_4bit_dora` | same |
| **AdaLoRA** | `SVDLinear4bit`, `test_4bit_adalora_causalLM` | `code\peft\src\peft\tuners\adalora\bnb.py` |
| **PiSSA** | `test_bloomz_pissa_4bit` | same |
| **OLoRA** | `test_bloomz_olora_4bit` | same |
| **LoftQ** | `loftq_utils_rationale_360` | `code\peft\src\peft\utils\loftq_utils.py` |
| **QALoRA (GPTQ)** | `qalora_gptq_finetuning_rationale_153` | `code\peft\examples\qalora_finetuning\qalora_gptq_finetuning.py` |

### Bottom line

**QLoRA** (NF4 4-bit quantization + LoRA via bitsandbytes + PEFT) is the de-facto standard for single-GPU fine-tuning. The 4-bit quantized base model drops VRAM ~4×, leaving room for LoRA adapter gradients within 24 GB. Extended variants (DoRA, PiSSA, LoftQ, AdaLoRA) all plug into the same `Linear4bit` path and are equally viable on a single 4090.

---

## Q05

**Pergunta:** How do DoRA, AdaLoRA, VeRA and LoRA+ improve over base LoRA?

**Modelo usado:** claude-sonnet (via `claude -p --model sonnet`) com a **skill graphify** (BFS subgraph traversal sobre `graphify-out/graph.json`)

### Stage 1 (papers só (54 arquivos · 118 nodes · 168 edges))

**Resposta:**

## How DoRA, AdaLoRA, VeRA, and LoRA+ improve over base LoRA

**Base LoRA** (`02-lora/01-lora-original.md`) freezes pretrained weights and adds low-rank update matrices ΔW = BA per layer. All four variants target distinct weaknesses of this design.

---

**DoRA** — *weight decomposition* (`02-lora/03-dora.md` · arxiv 2402.09353)
LoRA updates magnitude and direction of weights proportionally, leaving an accuracy gap vs. full fine-tuning. DoRA decomposes each pretrained weight into a magnitude scalar **m** and a direction matrix **V**, then applies LoRA only to the directional component: `W = (m+Δm) × (V₀+BA)/‖V₀+BA‖`. Separating the two degrees of freedom yields **better learning capacity and training stability** across LLaMA and LLaVA benchmarks with no inference overhead (weights merge like LoRA). — ICML 2024 Oral.

---

**AdaLoRA** — *adaptive rank allocation* (`02-lora/05-adalora.md` · arxiv 2303.10512)
LoRA spreads its parameter budget uniformly across all weight matrices, which is suboptimal. AdaLoRA parameterizes updates as SVD: ΔW = PΛQ, then dynamically prunes low-importance singular values using gradient-based importance scores. Layers that matter more receive higher effective rank. **Outperforms LoRA at equal budget, especially in low-budget regimes** (GLUE, SQuAD, text generation). — ICLR 2023.

---

**VeRA** — *shared frozen random matrices* (`02-lora/04-vera.md` · arxiv 2310.11454)
LoRA introduces separate A, B matrices per layer (~2·L·r·d parameters). VeRA uses a **single frozen random matrix pair shared across all layers**; only tiny per-layer scaling vectors (d, b) are trained: `ΔWₗ = diag(bₗ)B̄ diag(dₗ)Ā`. Result: **~10× fewer trainable parameters than LoRA**; at equal parameter count, outperforms LoRA by **+4 accuracy points**. — ICLR 2024.

---

**LoRA+** — *asymmetric learning rates* (`02-lora/06-lora-plus.md` · arxiv 2402.12354)
Standard LoRA trains A and B with the same learning rate, which is suboptimal for wide models because their optimal update scales differ with model width. LoRA+ sets **η_B = 16 × η_A** (derived from infinite-width network theory). The change costs nothing computationally and yields **1–2% accuracy gains and up to 2× convergence speedup**. — ICML 2024.

### Stage 2 (papers + docs lib peft/trl/bnb (131 arquivos · 294 nodes · 368 edges))

**Resposta:**

## How LoRA Variants Improve Over Base LoRA

**Base LoRA** (`02-lora/01-lora-original.md`, [arXiv:2106.09685](https://arxiv.org/abs/2402.09353)) freezes pretrained weights and injects trainable low-rank matrices A and B with a fixed rank `r` applied uniformly across layers.

---

**LoRA+** (`02-lora/06-lora-plus.md`, [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)) identifies that using the same learning rate for A and B is suboptimal due to their different roles in the product AB. It applies **asymmetric learning rates** — a higher rate for B than A — improving convergence speed and final accuracy with no extra parameters.

**DoRA** (`02-lora/03-dora.md`, [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)) decomposes weight updates into **magnitude and direction** components. The direction is updated via low-rank adaptation while magnitude is trained separately. This mimics full fine-tuning's learning dynamics more closely and narrows the gap between LoRA and full FT in quality-sensitive tasks.

**AdaLoRA** (`02-lora/05-adalora.md`, [arXiv:2303.10512](https://arxiv.org/abs/2303.10512)) replaces fixed uniform rank with **SVD-based parameterization** and adaptive rank allocation. It learns which weight matrices deserve higher rank by pruning singular values, concentrating capacity where it matters most — instead of wasting parameters on less important layers.

**VeRA** (`02-lora/04-vera.md`, [arXiv:2310.11454](https://arxiv.org/abs/2310.11454)) takes the opposite approach to parameter efficiency: it uses **frozen shared random matrices** across all layers and trains only small per-layer scaling vectors. This drastically cuts the number of trainable parameters (by up to 10×) at the cost of some flexibility.

---

**Pattern:** LoRA+ and DoRA improve *training dynamics*; AdaLoRA improves *rank allocation*; VeRA improves *parameter count*. All four are EXTRACTED extensions of base LoRA in the graph (community 5–6, confidence 1.0).

### Stage 3 (papers + docs lib + 1058 .py (1189 arquivos · 18.803 nodes · 66.935 edges))

**Resposta:**

## How DoRA, AdaLoRA, VeRA, and LoRA+ improve over base LoRA

**Base LoRA** adds a frozen pretrained weight `W₀` and a low-rank delta `ΔW = B·A` (B∈ℝᵐˣʳ, A∈ℝʳˣⁿ), training only B and A uniformly across all layers with the same rank r.

---

### DoRA — magnitude/direction decomposition
**Source:** `code/peft/src/peft/tuners/lora/dora.py`

DoRA decomposes each weight into a **learnable magnitude vector** m and a **direction component** `(W₀ + B·A) / ‖W₀ + B·A‖_c`. The output is `m · (W₀ + ΔW) / ‖W₀ + ΔW‖_c`. The column-wise norm is detached from the backward pass (see section 4.3 of paper `2402.09353`). This separates magnitude adaptation from directional adaptation, giving the model more expressive fine-tuning without extra rank budget.

---

### AdaLoRA — adaptive SVD-based rank allocation
**Source:** `code/peft/src/peft/tuners/adalora/config.py`, `adalora/layer.py`

AdaLoRA replaces `B·A` with a full SVD triplet `B · diag(E) · A`, where `E` holds learnable singular values. A three-phase schedule (warmup → reduction → fine-tuning) uses EMA-smoothed importance scores to **prune singular values** of less-important layers and reallocate that rank budget to layers where it matters more. Orthogonal regularization keeps U/V well-conditioned.

---

### VeRA — shared frozen random projections
**Source:** `code/peft/src/peft/tuners/vera/config.py`, `vera/layer.py`; paper `2310.11454`

VeRA shares a **single pair of frozen random matrices** (vera_A, vera_B) across all layers. Each layer only trains two tiny **scaling vectors** λ_b (out-dim) and λ_d (rank-dim): `ΔW = λ_b · B · diag(λ_d) · A`. This reduces trainable parameters from O(r·d) per layer to O(r + d), enabling much higher ranks (default r=256) with far fewer total parameters than LoRA.

---

### LoRA+ — differential learning rates for A and B
**Source:** `code/peft/src/peft/optimizers/loraplus.py`; paper `2402.12354`

No architectural change. LoRA+ assigns **separate learning rates** to lora_A (`lr`) and lora_B (`lr × loraplus_lr_ratio`, ratio ≥ 1). Theory motivates this because B initializes to zero and must learn task-specific features (needs higher lr), while A acts as a fixed projection. This simple change accelerates convergence especially on harder tasks.

---

## Q06

**Pergunta:** Explain the quantization stack used by QLoRA: NF4, Double Quantization, paged optimizers

**Modelo usado:** claude-sonnet (via `claude -p --model sonnet`) com a **skill graphify** (BFS subgraph traversal sobre `graphify-out/graph.json`)

### Stage 1 (papers só (54 arquivos · 118 nodes · 168 edges))

**Resposta:**

## QLoRA Quantization Stack: NF4, Double Quantization, Paged Optimizers

*(Sources: `06-quantization/01-qlora-quantization.md` and `02-lora/02-qlora.md`, both citing arXiv:2305.14314)*

QLoRA's core contribution is making 65B-parameter fine-tuning fit on a single 48 GB GPU via three layered innovations:

---

### 1. NF4 — 4-bit NormalFloat

LLM weights follow an approximately normal distribution after pre-training. NF4 exploits this by placing its 16 quantization levels at **equal-probability quantile boundaries** of N(0,1), making it information-theoretically optimal for that distribution — superior to INT4 or FP4 for typical model weights.

Each block of weights is normalized to `[-1, 1]` before encoding, so the 16 fixed levels (e.g., `[-1.0, -0.6962, … 0.7230, 1.0]`) are reused across all blocks.

---

### 2. Double Quantization (DQ)

NF4 uses one 32-bit floating-point **scaling constant per 64-weight block**, adding 0.5 bits/param of overhead. DQ quantizes those constants themselves down to 8-bit, shrinking the overhead to ~0.127 bits/param — a saving of **~0.37 bits/param**, bringing the effective cost from 4.5 to 4.37 bits/param.

---

### 3. Paged Optimizers

Gradient checkpointing recomputes activations on the backward pass, causing unpredictable **GPU memory spikes** that can exceed VRAM even when average usage is fine. Paged optimizers use NVIDIA's unified memory (CPU-GPU shared address space) to automatically evict optimizer states (e.g., Adam moments) to CPU RAM during spikes, then page them back in — functionally transparent to the training loop.

---

### Combined Effect

| Model | BF16 full FT | QLoRA (NF4+DQ+Paged) |
|-------|-------------|----------------------|
| 7B    | ~28 GB      | ~6 GB                |
| 65B   | ~260 GB     | ~48 GB               |

All three innovations are implemented in the `bitsandbytes` library via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`.

### Stage 2 (papers + docs lib peft/trl/bnb (131 arquivos · 294 nodes · 368 edges))

**Resposta:**

## QLoRA Quantization Stack

The graph's **"Quantization-Enabled Fine-Tuning Stack"** hyperedge (EXTRACTED, confidence 0.95, source: `06-quantization/01-qlora-quantization.md`) unifies five nodes: `QLoRA Quantization`, `NF4`, `Double Quantization (DQ)`, `bitsandbytes Library`, and `LoRA`.

### NF4 (NormalFloat 4-bit)

Node `NF4: NormalFloat 4-bit Quantization` (source: `https://arxiv.org/abs/2305.14314`) is **implemented by** `bitsandbytes Library` (EXTRACTED edge). NF4 is an information-theoretically optimal data type for normally distributed weights: it places quantization bins such that each bin holds an equal fraction of the weight distribution. Because pretrained LLM weights are approximately Gaussian, NF4 incurs less quantization error than uniform INT4 for the same bit-width. `bitsandbytes` exposes this as `Linear4bit` (community 4, `code/bitsandbytes/README.md`).

### Double Quantization (DQ)

Node `Double Quantization (DQ)` (source: `https://arxiv.org/abs/2305.14314`) is also **implemented by** `bitsandbytes Library` (EXTRACTED). DQ quantizes the quantization constants themselves: the first-pass NF4 constants are stored in FP32 (~0.5 bits/param overhead), and DQ requantizes them to 8-bit, reducing that overhead to ~0.127 bits/param — saving ~0.37 bits per parameter across the entire model.

### Paged Optimizers

The graph contains `8-bit Optimizers (bitsandbytes)` and `8-bit Optimizers Paper (ICLR 2022, arXiv:2110.02861)` nodes (source: `code/bitsandbytes/README.md`, link: `implements` EXTRACTED). QLoRA's paged optimizers use NVIDIA unified memory to automatically page optimizer states from GPU to CPU when the GPU is near OOM, then page them back for the gradient update step — eliminating OOM crashes during long-sequence training without degrading throughput.

### Integration Point

The `QLoRA Quantization` node **extends** `LoRA` (EXTRACTED) — LoRA adapters train in BF16 on top of a frozen NF4-quantized backbone, with the three mechanisms above jointly enabling 65B-parameter fine-tuning on a single 48 GB GPU.

**Sources cited:** `https://arxiv.org/abs/2305.14314` (QLoRA paper — NF4, DQ, paged optimizers), `https://huggingface.co/docs/bitsandbytes/main/en/index` (bitsandbytes, NF4/DQ implementation), `code/bitsandbytes/README.md` (8-bit optimizers), `06-quantization/01-qlora-quantization.md` (quantization stack hyperedge).

### Stage 3 (papers + docs lib + 1058 .py (1189 arquivos · 18.803 nodes · 66.935 edges))

**Resposta:**

## QLoRA Quantization Stack: NF4, Double Quantization, Paged Optimizers

**NF4 (NormalFloat 4-bit)**

NF4 is a 4-bit data type designed to be information-theoretically optimal for normally distributed weights. The graph traces it to a `create_normal_map()` function that builds a lookup table of 16 quantization bins with equal probability mass under a standard normal distribution — meaning each bin represents the same fraction of weight values. This is exposed via `quantize_nf4()` / `dequantize_nf4()` in bitsandbytes, with `LinearNF4` as the `nn.Module` wrapper that stores frozen base-model weights. Triton kernels (`quantize_nf4_blockwise_kernel`, `dequant_nf4_kernel`) implement the GPU path.

*Sources: `code\bitsandbytes\bitsandbytes\functional.py` L169, L861; `code\bitsandbytes\bitsandbytes\nn\modules.py` L676–677; `code\bitsandbytes\bitsandbytes\backends\triton\kernels_4bit.py`*

**Double Quantization**

The graph contains a dedicated `Double Quantization` node (`02-lora/02-qlora.md`, citing `https://arxiv.org/abs/2305.14314`) with the edge `[implements] → QLoRA`. Double Quantization refers to quantizing the quantization constants themselves: after each NF4 block has an fp32 scale factor, those scale factors are again quantized (to 8-bit blockwise), reducing the per-parameter memory overhead of the scales from ~0.5 bits/param to ~0.127 bits/param. The `QuantState` node (`functional.py` L419) is the runtime container that carries both the primary quantized tensor and these secondary quantized scales, used by `Params4bit`, `Linear4bit`, and `LinearNF4`.

*Sources: `02-lora/02-qlora.md` + `https://arxiv.org/abs/2305.14314`; `code\bitsandbytes\bitsandbytes\functional.py` L419*

**Paged Optimizers**

The graph shows a family of paged optimizer classes (`PagedAdam`, `PagedAdamW`, `PagedAdam8bit`, `PagedAdamW8bit`, `PagedAdamW32bit`, …) all inheriting from `Optimizer2State` in `bitsandbytes/optim/adamw.py` and `adam.py`. They use `get_paged()` (`functional.py` L90), which calls `cget_managed_ptr()` to allocate CUDA unified/managed memory. This lets optimizer states spill from GPU to CPU RAM during gradient spikes without crashing — the GPU pages them back in on demand. A benchmark rationale node (`examples/xpu/benchmark_paged_memory.py`) explicitly documents this GPU↔CPU paging behaviour.

*Sources: `code\bitsandbytes\bitsandbytes\optim\adamw.py` L179, L229; `code\bitsandbytes\bitsandbytes\functional.py` L90; `code\bitsandbytes\bitsandbytes\examples\xpu\benchmark_paged_memory.py`*

**How the three layers compose:** Base model weights are frozen and stored in NF4 via `LinearNF4` (≈4 bits/param); Double Quantization compresses the NF4 scale factors (saving ~0.37 bits/param); Paged Optimizers hold only the trainable LoRA adapter parameters but ensure that their Adam moment buffers never cause an OOM by offloading to CPU on demand. Together these three mechanisms allow fine-tuning a 65 B parameter model on a single consumer GPU.

---

## Q07

**Pergunta:** When to use DPO vs ORPO vs SimPO vs KTO for alignment?

**Modelo usado:** claude-sonnet (via `claude -p --model sonnet`) com a **skill graphify** (BFS subgraph traversal sobre `graphify-out/graph.json`)

### Stage 1 (papers só (54 arquivos · 118 nodes · 168 edges))

**Resposta:**

## When to use DPO vs ORPO vs SimPO vs KTO

All four methods sit in the same "preference optimization" hyperedge (`04-rlhf-dpo/05-dpo-survey-2024.md`, arxiv:2410.15595), but differ on two axes: **data format** and **pipeline stage**.

### DPO — the default baseline
Use when you have **paired preference data** (chosen/rejected per prompt) and a **pre-existing SFT checkpoint**. It eliminates the reward model and PPO but still requires a frozen reference model. Main risk: overfitting when the reference model is weak; sensitive to β tuning.
*Source: `04-rlhf-dpo/01-dpo-original.md` (arxiv:2305.18290)*

### KTO — when you lack preference pairs
Use when your dataset has only **binary desirability labels** (good/bad) rather than pairwise comparisons — common in production logs (thumbs up/down). Grounded in prospect theory; explicitly encodes **loss aversion** (penalty for bad outputs is weighted higher). Matches or exceeds DPO at ≥13B scale.
*Source: `04-rlhf-dpo/02-kto.md` (arxiv:2402.01306)*

### SimPO — reference-free, paired data, better benchmarks
Use when you want to **drop the reference model** while still training on preference pairs. Its length-normalized average log-probability reward eliminates length bias and reduces compute (no reference forward passes). Outperforms DPO by up to 7.5 pts on Arena-Hard; ranked #1 among <10B models on Chatbot Arena at release.
*Source: `04-rlhf-dpo/03-simpo.md` (arxiv:2405.14734)*

### ORPO — single-stage, training from scratch
Use when you want to **skip the separate SFT step** entirely. ORPO folds SFT (NLL loss) and preference alignment (odds ratio penalty) into one monolithic pass — saving ~50% compute and eliminating the reference model. Ideal when fine-tuning a base model with a preference dataset from the start.
*Source: `04-rlhf-dpo/04-orpo.md` (arxiv:2403.07691)*

---

**Decision summary:**

| Situation | Method |
|---|---|
| Have pairs + SFT model | DPO |
| Have only binary labels | KTO |
| Have pairs, want no ref model | SimPO |
| Training from base, single stage | ORPO |

### Stage 2 (papers + docs lib peft/trl/bnb (131 arquivos · 294 nodes · 368 edges))

**Resposta:**

## DPO vs ORPO vs SimPO vs KTO — When to Use Each

### Decision axis 1: Data format

**Your data has only binary labels** (good/bad per output, not paired comparisons) → **KTO**.
KTO replaces pairwise preference modeling with a prospect-theoretic utility function; it matches or exceeds DPO at 13B–30B scale and requires no rejected counterpart per prompt.
*Source: `04-rlhf-dpo/02-kto.md`, arxiv:2402.01306 (ICML 2024)*

---

### Decision axis 2: Training stage

**Starting a model from scratch** (no existing SFT checkpoint) → **ORPO**.
ORPO fuses SFT and preference alignment into a single loss (NLL + odds-ratio term), eliminating the two-stage SFT→DPO pipeline and the reference model entirely — cutting compute ~50%.
*Source: `04-rlhf-dpo/04-orpo.md`, arxiv:2403.07691*

---

### Decision axis 3: Reference model cost vs. quality

**Already have an SFT checkpoint, want best benchmark performance** → **SimPO**.
SimPO drops the reference model and replaces DPO's log-ratio reward with a length-normalized average log-probability + target margin γ. It outperforms DPO by up to 7.5 pts on Arena-Hard and corrects DPO's length bias.
*Source: `04-rlhf-dpo/03-simpo.md`, arxiv:2405.14734 (NeurIPS 2024)*

**Standard baseline / existing infra already runs a reference model** → **DPO**.
DPO is the theoretically grounded foundation (Bradley-Terry, closed-form RL solution). Use it when you want a well-understood method, have paired data, and reference model memory is not a constraint. Be aware: it can overfit when the reference model is weak, and it favors longer responses.
*Source: `04-rlhf-dpo/05-dpo-survey-2024.md`, arxiv:2410.15595*

---

### Summary table

| Condition | Method |
|---|---|
| Only binary (thumbs up/down) labels | **KTO** |
| No SFT checkpoint yet, want one-shot training | **ORPO** |
| Have SFT checkpoint, want max performance | **SimPO** |
| Have SFT checkpoint + reference model, want a solid baseline | **DPO** |

### Stage 3 (papers + docs lib + 1058 .py (1189 arquivos · 18.803 nodes · 66.935 edges))

**Resposta:**

## DPO vs ORPO vs SimPO vs KTO: When to Use Each

Based on the **"Preference Optimization Methods"** hyperedge (confidence 0.90) and the four source nodes in the graph:

---

### DPO — the default baseline
**Use when:** you have **paired preference data** (chosen/rejected per prompt), you've already run SFT, and you want a stable, well-understood approach with a reference model available.

DPO solves KL-constrained reward maximization in closed form, eliminating PPO/reward model. Requires offline data and a reference model (frozen copy of the SFT model). Prone to length bias.

*Source: `04-rlhf-dpo/01-dpo-original.md` (arxiv 2305.18290)*

---

### KTO — when you only have binary labels
**Use when:** your data has **unpaired, binary desirability labels** ("good"/"bad") rather than pairwise preferences. Especially useful when you can't construct (chosen, rejected) pairs — e.g., logs of individual completions with thumbs-up/down.

KTO matches or beats DPO at 13B+ scales and is grounded in prospect theory / loss aversion. Still requires a reference model.

*Source: `04-rlhf-dpo/02-kto.md` (arxiv 2402.01306)*

---

### ORPO — when you want to skip the SFT stage
**Use when:** you are training **from a base/instruct model directly** and want to combine SFT + alignment in **one pass**. Eliminates the reference model and the two-stage SFT→DPO pipeline, cutting compute ~50%.

ORPO folds an odds-ratio penalty into the NLL loss. Requires paired data (chosen/rejected) but no separate reference model.

*Source: `04-rlhf-dpo/04-orpo.md` (arxiv 2403.07691)*

---

### SimPO — when you want best benchmark performance without a reference model
**Use when:** you want **top alignment performance** with **no reference model overhead** and have paired preference data. SimPO's length-normalized average log-prob reward removes DPO's length bias.

Outperforms DPO by up to 7.5 pts (Arena-Hard) and achieves state-of-the-art on AlpacaEval 2 for <10B models. Adds a target reward margin γ to ensure a meaningful gap between chosen/rejected.

*Source: `04-rlhf-dpo/03-simpo.md` (arxiv 2405.14734)*

---

### Quick decision guide

| Situation | Use |
|---|---|
| Standard paired data, ref model available | **DPO** |
| Binary labels only (no pairs) | **KTO** |
| Single-stage training (no separate SFT run) | **ORPO** |
| Paired data, no ref model, max perf | **SimPO** |

---
