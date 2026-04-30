# Graph Report - .  (2026-04-30)

## Corpus Check
- 54 files · ~26,518 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 118 nodes · 168 edges · 11 communities detected
- Extraction: 68% EXTRACTED · 32% INFERRED · 0% AMBIGUOUS · INFERRED: 54 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_PEFT Methods Survey|PEFT Methods Survey]]
- [[_COMMUNITY_Continual Learning & Forgetting|Continual Learning & Forgetting]]
- [[_COMMUNITY_Reasoning RL & Preference|Reasoning RL & Preference]]
- [[_COMMUNITY_Instruction Tuning & Long Context|Instruction Tuning & Long Context]]
- [[_COMMUNITY_Preference Optimization & Distillation|Preference Optimization & Distillation]]
- [[_COMMUNITY_LoRA Family & Adapters|LoRA Family & Adapters]]
- [[_COMMUNITY_Distillation & GRPO|Distillation & GRPO]]
- [[_COMMUNITY_Quantization & Domain SFT|Quantization & Domain SFT]]
- [[_COMMUNITY_Quantization Methods|Quantization Methods]]
- [[_COMMUNITY_Data Selection & Multilingual|Data Selection & Multilingual]]
- [[_COMMUNITY_Long Context Methods|Long Context Methods]]

## God Nodes (most connected - your core abstractions)
1. `LoRA: Low-Rank Adaptation` - 11 edges
2. `LoRA (Low-Rank Adaptation)` - 8 edges
3. `DoRA (Weight-Decomposed Low-Rank Adaptation)` - 8 edges
4. `TRL SFTTrainer` - 8 edges
5. `Direct Preference Optimization (DPO)` - 7 edges
6. `Parameter-Efficient Fine-Tuning (PEFT)` - 7 edges
7. `Model Merging for Continual Learning (TIES, DARE, Task Arithmetic)` - 6 edges
8. `Instruction Tuning (SFT)` - 5 edges
9. `TRL â€” Transformers Reinforcement Learning Library` - 5 edges
10. `QLoRA` - 5 edges

## Surprising Connections (you probably didn't know these)
- `FLawN-T5: Legal Instruction-Tuned Model` --semantically_similar_to--> `TRL SFTTrainer`  [INFERRED] [semantically similar]
  08-domain-specific/04-lawinstruct-legal.md → 01-sft/03-trl-sft-trainer-docs.md
- `Block Expansion Method (Identity-Initialized Blocks)` --semantically_similar_to--> `I-LoRA (Interpolation-based LoRA for Continual Learning)`  [INFERRED] [semantically similar]
  03-adapters-prompts/06-llama-pro-block-expansion.md → 10-continual/02-catastrophic-forgetting-peft.md
- `LongRoPE (Non-Uniform RoPE Interpolation)` --semantically_similar_to--> `FP8-LM Training Framework`  [INFERRED] [semantically similar]
  07-long-context/02-longrope.md → 06-quantization/03-fp8-lm-training.md
- `Prefix-Tuning` --semantically_similar_to--> `LoRA: Low-Rank Adaptation`  [INFERRED] [semantically similar]
  03-adapters-prompts/01-prefix-tuning.md → 02-lora/01-lora-original.md
- `Evol-Instruct` --semantically_similar_to--> `Self-Instruct Paradigm`  [INFERRED] [semantically similar]
  08-domain-specific/02-wizardcoder.md → 01-sft/01-instruction-tuning-survey.md

## Hyperedges (group relationships)
- **LoRA Variant Family** — 08_lora_learns_lora, 02_qlora_qlora, 05_adalora_adalora, 05_peft_adapters [INFERRED 0.90]
- **Preference Optimization Methods** — 05_dpo_dpo, 02_kto_kto, 05_dpo_simpo, 05_dpo_orpo [INFERRED 0.90]
- **Long Context Extension Methods** — 04_position_interpolation, 01_yarn, 04_rope [INFERRED 0.85]
- **Continual Learning Strategies** — 01_continual_survey, 04_model_merging, 01_continual_ewc, 08_lora_learns_lora [INFERRED 0.80]
- **TRL Trainer Ecosystem** — 04_trl_library, 04_trl_sft_trainer, 04_trl_dpo_trainer, 04_trl_grpo_trainer_ref, 04_trl_kto_trainer, 04_grpo_trainer_docs [EXTRACTED 1.00]
- **LoRA Variant Family** — 03_dora_dora, 06_lora_plus_lora_plus, 02_cf_peft_i_lora, 09_lora_survey_vera, 09_lora_survey_lora_merging [INFERRED 0.90]
- **PEFT Taxonomy Classes** — 02_peft_survey_additive_peft, 02_peft_survey_selective_peft, 02_peft_survey_reparameterized_peft, 02_peft_survey_hybrid_peft [EXTRACTED 1.00]
- **Preference Optimization Methods** — 06_trl_dpo_dpo, 03_simpo_simpo, 02_deepseekmath_grpo [INFERRED 0.80]
- **Long Context Position Embedding Methods** — 02_longrope_longrope, 05_long_ctx_medical_yarn, 05_long_ctx_medical_longloRA [EXTRACTED 0.90]
- **On-Policy Distillation Paradigm** — 01_minillm_minillm, 04_opd_survey_opd_survey, 05_open_r1_open_r1, 01_minillm_reverse_kld [INFERRED 0.80]
- **LoRA Family of Parameter-Efficient Fine-Tuning Methods** — 01_lora_original_lora, 04_vera_vera, 03_longlora_longlora, 03_curlora_curlora, 01_qlora_quantization_qlora [EXTRACTED 0.95]
- **PEFT Adapter Paradigm (pre-LoRA)** — 04_adapters_houlsby, 01_prefix_tuning, 04_adapters_bottleneck, 01_lora_original_lora [INFERRED 0.85]
- **Preference Alignment Methods** — 01_dpo_original_dpo, 04_orpo_orpo, 07_llama2_rlhf, 03_tulu3_rlvr [INFERRED 0.90]
- **Domain-Specific SFT (Medical & Legal)** — 01_biomistral_biomistral, 04_lawinstruct_lawinstruct, 04_lawinstruct_flawnt5, 03_trl_sft_trainer [INFERRED 0.80]
- **Quantization-Enabled Fine-Tuning Stack** — 01_qlora_quantization_qlora, 01_qlora_nf4, 01_qlora_double_quantization, 04_bitsandbytes_lib, 01_lora_original_lora [EXTRACTED 0.95]

## Communities

### Community 0 - "PEFT Methods Survey"
Cohesion: 0.17
Nodes (18): I-LoRA (Interpolation-based LoRA for Continual Learning), Mode Connectivity in LoRA Fine-tuning, Additive PEFT, Hybrid PEFT, Parameter-Efficient Fine-Tuning (PEFT), Reparameterized PEFT, Selective PEFT, DoRA (Weight-Decomposed Low-Rank Adaptation) (+10 more)

### Community 1 - "Continual Learning & Forgetting"
Cohesion: 0.19
Nodes (16): Catastrophic Forgetting in LLMs, Elastic Weight Consolidation (EWC), Continual Learning of LLMs Survey 2024, Deep Prompt Tuning, P-Tuning v2, Model Merging for Continual Learning (TIES, DARE, Task Arithmetic), DARE (Drop and Rescale), Task Arithmetic (+8 more)

### Community 2 - "Reasoning RL & Preference"
Cohesion: 0.19
Nodes (15): Group Relative Policy Optimization (GRPO), DeepSeek-R1, Reinforcement Learning with Verifiable Rewards (RLVR), KTO (Kahneman-Tversky Optimization), Prospect Theory for Alignment, GRPO Trainer Documentation (TRL), DPOTrainer, GRPOTrainer (TRL Overview) (+7 more)

### Community 3 - "Instruction Tuning & Long Context"
Cohesion: 0.21
Nodes (13): Instruction Tuning for LLMs: A Survey, Instruction Tuning (SFT), Self-Instruct Paradigm, YaRN (Yet Another RoPE extensioN), NTK-by-Parts Interpolation, WizardCoder, Evol-Instruct, Black-box Knowledge Distillation (+5 more)

### Community 4 - "Preference Optimization & Distillation"
Cohesion: 0.22
Nodes (13): KL-Constrained Reward Maximization Closed Form, DPO: Direct Preference Optimization, Distribution Mismatch / Exposure Bias in KD, GKD: Generalized Knowledge Distillation, Token-level Cross-Entropy Loss (SFT), TRL SFTTrainer, RLVR: Reinforcement Learning with Verifiable Rewards, Tulu 3 Post-Training Pipeline (+5 more)

### Community 5 - "LoRA Family & Adapters"
Cohesion: 0.26
Nodes (12): Intrinsic Rank Hypothesis, LoRA: Low-Rank Adaptation, Prefix-Tuning, CUR Matrix Decomposition, CURLoRA: CUR Decomposition LoRA for Continual Learning, LongLoRA: Long-Context LoRA Fine-tuning, Shifted Sparse Attention (S2-Attn), Bottleneck Adapter Architecture (+4 more)

### Community 6 - "Distillation & GRPO"
Cohesion: 0.27
Nodes (11): MiniLLM (On-Policy Distillation with Reverse KLD), Reverse KL Divergence for Distillation, DeepSeekMath 7B, GRPO (Group Relative Policy Optimization), SimPO Length-Normalized Reward, SimPO (Simple Preference Optimization), f-Divergence Unification Framework (OPD), On-Policy Distillation Survey 2026 (+3 more)

### Community 7 - "Quantization & Domain SFT"
Cohesion: 0.38
Nodes (7): BioMistral: Medical Domain LLM, Double Quantization (DQ), NF4: NormalFloat 4-bit Quantization, QLoRA Quantization, bitsandbytes Library, FLawN-T5: Legal Instruction-Tuned Model, LawInstruct: Legal Domain Instruction Dataset

### Community 8 - "Quantization Methods"
Cohesion: 0.4
Nodes (6): EfficientQAT, Block-wise QAT Training, Double Quantization, Guanaco Model, NF4 (4-bit NormalFloat), QLoRA

### Community 9 - "Data Selection & Multilingual"
Cohesion: 0.5
Nodes (4): Multilingual Instruction Tuning (Pinch of Multilinguality), DEITA Complexity Scoring, DEITA (Data-Efficient Instruction Tuning for Alignment), DEITA Diversity Constraint

### Community 10 - "Long Context Methods"
Cohesion: 1.0
Nodes (3): LongRoPE (Non-Uniform RoPE Interpolation), LongLoRA (Sparse Attention for Long Context), YaRN (Non-Uniform Position Interpolation)

## Knowledge Gaps
- **27 isolated node(s):** `Double Quantization`, `Guanaco Model`, `SVD-Based Parameterization (AdaLoRA)`, `Deep Prompt Tuning`, `BitFit (Bias-Only Fine-Tuning)` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Medical Fine-Tuning Long-Context Study` connect `Instruction Tuning & Long Context` to `Continual Learning & Forgetting`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Why does `Instruction Tuning (SFT)` connect `Instruction Tuning & Long Context` to `Reasoning RL & Preference`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `Catastrophic Forgetting in LLMs` connect `Continual Learning & Forgetting` to `Instruction Tuning & Long Context`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `LoRA: Low-Rank Adaptation` (e.g. with `Prefix-Tuning` and `Shifted Sparse Attention (S2-Attn)`) actually correct?**
  _`LoRA: Low-Rank Adaptation` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `LoRA (Low-Rank Adaptation)` (e.g. with `Catastrophic Forgetting in LLMs` and `Adapter Methods (PEFT)`) actually correct?**
  _`LoRA (Low-Rank Adaptation)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `DoRA (Weight-Decomposed Low-Rank Adaptation)` (e.g. with `Reparameterized PEFT` and `LoRA+ (Asymmetric Learning Rates)`) actually correct?**
  _`DoRA (Weight-Decomposed Low-Rank Adaptation)` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `TRL SFTTrainer` (e.g. with `Tulu 3 Post-Training Pipeline` and `BioMistral: Medical Domain LLM`) actually correct?**
  _`TRL SFTTrainer` has 4 INFERRED edges - model-reasoned connections that need verification._