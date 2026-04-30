# Graph Report - .  (2026-04-30)

## Corpus Check
- 131 files · ~50,000 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 294 nodes · 368 edges · 18 communities detected
- Extraction: 76% EXTRACTED · 24% INFERRED · 0% AMBIGUOUS · INFERRED: 89 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]

## God Nodes (most connected - your core abstractions)
1. `LoraConfig` - 18 edges
2. `LoRA: Low-Rank Adaptation` - 11 edges
3. `PeftModel` - 10 edges
4. `TRL Library (Transformers Reinforcement Learning)` - 10 edges
5. `GRPOTrainer` - 9 edges
6. `LoRA (Low-Rank Adaptation)` - 8 edges
7. `DoRA (Weight-Decomposed Low-Rank Adaptation)` - 8 edges
8. `TRL SFTTrainer` - 8 edges
9. `LoraConfig` - 8 edges
10. `Direct Preference Optimization (DPO)` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Block Expansion Method (Identity-Initialized Blocks)` --semantically_similar_to--> `I-LoRA (Interpolation-based LoRA for Continual Learning)`  [INFERRED] [semantically similar]
  03-adapters-prompts/06-llama-pro-block-expansion.md → 10-continual/02-catastrophic-forgetting-peft.md
- `FLawN-T5: Legal Instruction-Tuned Model` --semantically_similar_to--> `TRL SFTTrainer`  [INFERRED] [semantically similar]
  08-domain-specific/04-lawinstruct-legal.md → 01-sft/03-trl-sft-trainer-docs.md
- `Prefix-Tuning` --semantically_similar_to--> `LoRA: Low-Rank Adaptation`  [INFERRED] [semantically similar]
  03-adapters-prompts/01-prefix-tuning.md → 02-lora/01-lora-original.md
- `Evol-Instruct` --semantically_similar_to--> `Self-Instruct Paradigm`  [INFERRED] [semantically similar]
  08-domain-specific/02-wizardcoder.md → 01-sft/01-instruction-tuning-survey.md
- `Black-box Knowledge Distillation` --semantically_similar_to--> `Self-Instruct Paradigm`  [INFERRED] [semantically similar]
  09-distillation/03-kd-survey-llm-2024.md → 01-sft/01-instruction-tuning-survey.md

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
- **TRL Trainer Family** — trl_sfttrainer, trl_dpotrainer, trl_grpotrainer, trl_asyncgrpotrainer, trl_rewardtrainer, trl_ktotrainer, trl_onlinedpotrainer, trl_ppotrainer, trl_xpotrainer, trl_gkdtrainer, trl_minillmtrainer, trl_sdpotrainer, trl_ssdtrainer [EXTRACTED 1.00]
- **PEFT Config Hierarchy** — peft_loraconfig, peft_lohaconfig, peft_lokrconfig, peft_adaloraconfig [EXTRACTED 1.00]
- **bitsandbytes Quantization Family** — bnb_linear8bitlt, bnb_linear4bit, bnb_llmint8_concept, bnb_qlora_concept [EXTRACTED 1.00]
- **TRL Knowledge Distillation Trainers** — trl_gkdtrainer, trl_minillmtrainer, trl_sdpotrainer, trl_ssdtrainer [EXTRACTED 1.00]
- **TRL Online RL Trainers** — trl_grpotrainer, trl_asyncgrpotrainer, trl_ppotrainer, trl_onlinedpotrainer, trl_xpotrainer [EXTRACTED 1.00]
- **PEFT LoRA Variants** — peft_loraconfig, peft_lohaconfig, peft_lokrconfig, peft_adaloraconfig, peft_lora_paper, peft_loha_paper, peft_lokr_paper, peft_adalora_paper [EXTRACTED 1.00]
- **TRL Trainer Family** — sft_trainer_sfttrainer, dpo_trainer_dpotrainer, rloo_trainer_rlootrainer, nash_md_trainer_nashmdftrainer, bco_trainer_bcotrainer, prm_trainer_prmtrainer, tpo_trainer_tpotrainer, gold_trainer_goldtrainer, papo_trainer_papotrainer, openenv_grpotrainer_environments [EXTRACTED 0.95]
- **PEFT Config Hierarchy** — lora_loraconfig, ia3_ia3config, prompt_methods_prompt_tuning_config, prompt_methods_prefix_tuning_config, prompt_methods_prompt_encoder_config [EXTRACTED 0.95]
- **LoRA Initialization Methods** — lora_pissa, lora_eva, lora_olora, lora_corda, lora_rslora, lora_loftq, lora_loraga [EXTRACTED 0.95]
- **Memory-Efficient Training Methods** — deepspeed_qlora_deepspeed, deepspeed_zero3_stage, liger_kernel_liger_kernel_integration, compile_h100_l40_bitsandbytes_compile_guide [INFERRED 0.85]
- **TRL Dataset Types** — dataset_formats_preference_dataset, dataset_formats_unpaired_preference_dataset, dataset_formats_stepwise_supervision_dataset, dataset_formats_prompt_only_dataset [EXTRACTED 0.98]
- **Model Merging Methods (PEFT)** — model_merging_ties, model_merging_dare, model_merging_add_weighted_adapter, lora_merge_and_unload [EXTRACTED 0.95]
- **TRL Trainer Classes** — grpo_grpotrainer, orpo_orpotrainer, cpo_cpotrainer, distillation_distillationtrainer, sdft_sdfttrainer [EXTRACTED 0.95]
- **TRL+PEFT Integration** — peft_peftmodel, peft_get_peft_model, peft_integration_qlora, peft_integration_prompt_tuning, fsdp_qlora_bitsandbytesconfig, grpo_grpotrainer, orpo_orpotrainer, cpo_cpotrainer [EXTRACTED 0.95]
- **vLLM+TRL Integration** — vllm_integration_vllm_server, vllm_integration_pagedattention, grpo_grpotrainer, distillation_distillationtrainer [EXTRACTED 0.95]
- **Quantization Methods** — fsdp_qlora_bitsandbytesconfig, quantization_aqlm, quantization_hqq, quantization_gptq, quantization_loftq, peft_integration_qlora [EXTRACTED 0.90]
- **PEFT Adapter Methods** — oft_oftconfig, oft_boftconfig, ia3_ia3config, peft_integration_prompt_tuning, peft_peftmodel [EXTRACTED 0.90]
- **GRPO Loss Variants** — grpo_group_relative_policy_optimization, grpo_dapo, grpo_sapo, grpo_dr_grpo [EXTRACTED 0.95]
- **FSDP-QLoRA Training Stack** — fsdp_qlora_fsdp_qlora, fsdp_qlora_bitsandbytesconfig, fsdp_qlora_linear4bit, fsdp_peft_fsdp_auto_wrap_policy, peft_peftmodel [EXTRACTED 0.95]
- **Memory Reduction Techniques** — reducing_memory_liger_kernel, reducing_memory_gradient_checkpointing, peft_integration_qlora, vllm_integration_vllm_server, kernels_hub_flash_attn2 [EXTRACTED 0.90]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.1
Nodes (32): BioMistral: Medical Domain LLM, KL-Constrained Reward Maximization Closed Form, DPO: Direct Preference Optimization, Intrinsic Rank Hypothesis, LoRA: Low-Rank Adaptation, Prefix-Tuning, Double Quantization (DQ), NF4: NormalFloat 4-bit Quantization (+24 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (28): Group Relative Policy Optimization (GRPO), DeepSeek-R1, Reinforcement Learning with Verifiable Rewards (RLVR), Instruction Tuning for LLMs: A Survey, Instruction Tuning (SFT), Self-Instruct Paradigm, YaRN (Yet Another RoPE extensioN), NTK-by-Parts Interpolation (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (26): LoraLayer (PEFT internal class), PEFT Checkpoint Format (adapter_model.safetensors + adapter_config.json), bitsandbytes H100/L40 Compile Guide, BNB_CUDA_VERSION environment variable, H100 GPU sm_90 (Compute Capability 9.0), L40 GPU sm_89 (Compute Capability 8.9), PEFT + DeepSpeed Integration, QLoRA + DeepSpeed ZeRO3 Configuration (+18 more)

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (26): BCOConfig, BCOTrainer (Binary Classifier Optimization), Preference Dataset (chosen/rejected), Prompt-Only Dataset, Stepwise Supervision Dataset (process reward), Unpaired Preference Dataset (completion + label), DPO Paper - Direct Preference Optimization (Rafailov et al. 2023), DPOTrainer (+18 more)

### Community 4 - "Community 4"
Cohesion: 0.1
Nodes (24): Linear4bit, QLoRA Paper (arXiv:2305.14314), QLoRA 4-bit Quantization, Parameter-Efficient Fine-Tuning (PEFT), Reinforcement Learning from Human Feedback (RLHF), AdaLoRA Paper (arXiv:2303.10512), AdaLoraConfig, BOFT Paper (arXiv:2311.06243) (+16 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (22): Catastrophic Forgetting in LLMs, Elastic Weight Consolidation (EWC), Continual Learning of LLMs Survey 2024, EfficientQAT, Block-wise QAT Training, Deep Prompt Tuning, P-Tuning v2, Double Quantization (+14 more)

### Community 6 - "Community 6"
Cohesion: 0.14
Nodes (21): I-LoRA (Interpolation-based LoRA for Continual Learning), Mode Connectivity in LoRA Fine-tuning, LongRoPE (Non-Uniform RoPE Interpolation), Additive PEFT, Hybrid PEFT, Parameter-Efficient Fine-Tuning (PEFT), Reparameterized PEFT, Selective PEFT (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (20): Chat Template (Jinja2), get_training_chat_template, DistillationConfig, DistillationTrainer, Generalized Knowledge Distillation (GKD), DAPO (token-level normalization), Dr. GRPO (length-bias-free), Group Relative Policy Optimization (GRPO) (+12 more)

### Community 8 - "Community 8"
Cohesion: 0.12
Nodes (19): AutoPeftModelForSeq2SeqLM, IA3 - Infused Adapter by Inhibiting and Amplifying Inner Activations, IA3Config, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict, get_peft_model(), PromptTuningConfig (+11 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (17): Knowledge Distillation for LLMs, Supervised Fine-Tuning (SFT), AsyncGRPOTrainer, GKD Paper (arXiv:2306.13649), GKDConfig, GKDTrainer, GRPO Paper: DeepSeekMath (arXiv:2402.03300), GRPOConfig (+9 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (15): fsdp_auto_wrap_policy, BitsAndBytesConfig, FSDP-QLoRA, nn.Linear4bit, nn.Params4bit, LoRA Rank Guidelines (LoRA Without Regret), Butterfly OFT (BOFT), BOFTConfig (+7 more)

### Community 11 - "Community 11"
Cohesion: 0.17
Nodes (12): Linear8bitLt, LLM.int8() Quantization, LLM.int8() Paper (arXiv:2208.07339), Direct Preference Optimization (DPO), DPOConfig, DPOTrainer, KTO Paper (arXiv:2402.01306), KTOTrainer (+4 more)

### Community 12 - "Community 12"
Cohesion: 0.27
Nodes (11): MiniLLM (On-Policy Distillation with Reverse KLD), Reverse KL Divergence for Distillation, DeepSeekMath 7B, GRPO (Group Relative Policy Optimization), SimPO Length-Normalized Reward, SimPO (Simple Preference Optimization), f-Divergence Unification Framework (OPD), On-Policy Distillation Survey 2026 (+3 more)

### Community 13 - "Community 13"
Cohesion: 0.32
Nodes (8): AlphaPO, CPOConfig, CPOTrainer, SimPO (Simple Preference Optimization), Odds Ratio Preference Optimization (ORPO), ORPOConfig, ORPOTrainer, DiagnosticsCallback (PTT)

### Community 14 - "Community 14"
Cohesion: 0.5
Nodes (4): Multilingual Instruction Tuning (Pinch of Multilinguality), DEITA Complexity Scoring, DEITA (Data-Efficient Instruction Tuning for Alignment), DEITA Diversity Constraint

### Community 15 - "Community 15"
Cohesion: 0.5
Nodes (4): Soft Prompts / Prompt Tuning Methods, Prefix Tuning Paper (arXiv:2101.00190), Prompt Tuning Paper (arXiv:2104.08691), P-Tuning Paper (arXiv:2103.10385)

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (2): 8-bit Optimizers (bitsandbytes), 8-bit Optimizers Paper (ICLR 2022, arXiv:2110.02861)

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (2): AdEMAMix Optimizer, AdEMAMix Paper (arXiv:2409.03137)

## Knowledge Gaps
- **103 isolated node(s):** `Double Quantization`, `Guanaco Model`, `SVD-Based Parameterization (AdaLoRA)`, `Deep Prompt Tuning`, `BitFit (Bias-Only Fine-Tuning)` (+98 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 16`** (2 nodes): `8-bit Optimizers (bitsandbytes)`, `8-bit Optimizers Paper (ICLR 2022, arXiv:2110.02861)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (2 nodes): `AdEMAMix Optimizer`, `AdEMAMix Paper (arXiv:2409.03137)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LoraConfig` connect `Community 2` to `Community 8`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `get_peft_model()` connect `Community 8` to `Community 10`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `LoraConfig` (e.g. with `LoraLayer (PEFT internal class)` and `LoraModel.add_weighted_adapter()`) actually correct?**
  _`LoraConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `LoRA: Low-Rank Adaptation` (e.g. with `Prefix-Tuning` and `Shifted Sparse Attention (S2-Attn)`) actually correct?**
  _`LoRA: Low-Rank Adaptation` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `TRL Library (Transformers Reinforcement Learning)` (e.g. with `PAPOTrainer (Perception-Aware Policy Optimization)` and `GRPOTrainer + OpenEnv (RL environments with state)`) actually correct?**
  _`TRL Library (Transformers Reinforcement Learning)` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Double Quantization`, `Guanaco Model`, `SVD-Based Parameterization (AdaLoRA)` to the rest of the system?**
  _103 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._