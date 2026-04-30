# Queries — Fine-Tuning para low-GPU: LoRA vs QLoRA vs Full FT

Lista de queries pra rodar nas 2 ferramentas durante a apresentação. Organizadas por eixo do problema. Cada query lista **ferramenta**, **comando exato**, e **insight esperado**.

---

## 1. Comparação direta: LoRA vs QLoRA vs Full Fine-Tuning

### 1.1 Visão geral comparativa
- **graphify** — mostra como os métodos se conectam:
  ```bash
  graphify query "When to use LoRA vs QLoRA vs full fine-tuning for low-GPU devices"
  ```
  Vai retornar subgrafo BFS com god nodes (LoRA, QLoRA, Full FT) + comunidades adjacentes (Quantization, PEFT Methods).

- **edgequake (UI ou API)** — query natural language:
  ```
  POST /api/v1/query  { "query": "Compare LoRA, QLoRA and full fine-tuning trade-offs for memory-constrained settings", "mode": "naive", "top_k": 8 }
  ```

### 1.2 Caminhos de evolução
- **graphify path** — relação direta entre métodos:
  ```bash
  graphify path "LoRA" "QLoRA"
  graphify path "Full Fine-Tuning" "QLoRA"
  graphify path "QLoRA" "DoRA"
  ```
  Mostra hops e relações (`extends`, `inspired_by`, `cites`) que conectam.

### 1.3 Explicar nó central
- **graphify explain**:
  ```bash
  graphify explain "QLoRA"
  graphify explain "LoRA"
  graphify explain "DoRA"
  graphify explain "AdaLoRA"
  ```

---

## 2. Eixo: Consumo de VRAM

### 2.1
- **edgequake**:
  ```
  "What is the VRAM consumption of LoRA QLoRA and full fine-tuning on a 13B model?"
  ```
- **graphify**:
  ```bash
  graphify query "VRAM memory consumption of QLoRA NF4 quantization vs full precision fine-tuning"
  ```

### 2.2 Quantization stack
- **graphify path**:
  ```bash
  graphify path "QLoRA" "bitsandbytes"
  graphify path "NF4" "Double Quantization"
  ```

### 2.3 Hyperedge "Quantization-Enabled Fine-Tuning Stack"
- **graphify explain**:
  ```bash
  graphify explain "Quantization-Enabled Fine-Tuning Stack"
  graphify explain "NF4"
  graphify explain "Double Quantization"
  ```

---

## 3. Eixo: Qualidade final

### 3.1
- **edgequake**:
  ```
  "How does LoRA quality compare to full fine-tuning on benchmarks like MMLU or GSM8K?"
  "Does QLoRA degrade performance compared to LoRA?"
  "What is the gap in downstream task accuracy between PEFT methods and full fine-tuning?"
  ```
- **graphify**:
  ```bash
  graphify query "Quality degradation tradeoff LoRA vs full fine-tuning on benchmarks"
  graphify query "DoRA improvement over LoRA accuracy"
  ```

### 3.2 Quando LoRA "esquece menos"
- **edgequake**:
  ```
  "Does LoRA suffer less from catastrophic forgetting than full fine-tuning?"
  ```
- **graphify**:
  ```bash
  graphify explain "LoRA Learns Less Forgets Less"
  ```

---

## 4. Eixo: Custo computacional / velocidade de treino

### 4.1
- **edgequake**:
  ```
  "What is the training speed difference between LoRA, QLoRA and full fine-tuning?"
  "How does FP8 training affect throughput compared to FP16 or BF16?"
  ```
- **graphify**:
  ```bash
  graphify query "Training time and FLOPs reduction with PEFT methods"
  graphify path "FP8 Training" "QLoRA"
  ```

### 4.2 GPU compute requirements
- **edgequake**:
  ```
  "Minimum GPU requirements to fine-tune a 7B 13B and 70B model with QLoRA"
  ```

---

## 5. Eixo: Hardware requirements

### 5.1
- **edgequake**:
  ```
  "Which fine-tuning method works on a single RTX 4090 24GB?"
  "Can a 7B model be fine-tuned on consumer hardware with QLoRA?"
  ```
- **graphify**:
  ```bash
  graphify query "Single GPU fine-tuning of large language models with quantization"
  graphify explain "bitsandbytes Library"
  ```

### 5.2 Edge / TinyML angle (caso queira esticar pro tema Edge AI)
- **edgequake**:
  ```
  "How can fine-tuned models be deployed on edge devices?"
  "What is the smallest quantization that preserves accuracy?"
  ```

---

## 6. Métodos adjacentes (DoRA, AdaLoRA, VeRA, LoRA+, etc.)

### 6.1 Família LoRA — overview
- **graphify explain**:
  ```bash
  graphify explain "LoRA Variant Family"
  graphify explain "LoRA Family of Parameter-Efficient Fine-Tuning Methods"
  ```

### 6.2 Variantes específicas
- **graphify path**:
  ```bash
  graphify path "LoRA" "DoRA"
  graphify path "LoRA" "AdaLoRA"
  graphify path "LoRA" "VeRA"
  graphify path "LoRA" "LoRA+"
  ```

### 6.3 Comparação direta
- **edgequake**:
  ```
  "What are the differences between LoRA, DoRA, AdaLoRA, VeRA and LoRA+?"
  "Which LoRA variant gives best accuracy improvement over base LoRA?"
  ```

---

## 7. Preference optimization (DPO/ORPO/SimPO/KTO) — se quiser ampliar

### 7.1 Família
- **graphify explain**:
  ```bash
  graphify explain "Preference Optimization Methods"
  graphify explain "Preference Alignment Methods"
  ```

### 7.2 Comparação
- **edgequake**:
  ```
  "When to use DPO vs ORPO vs SimPO vs KTO for alignment?"
  ```

---

## 8. RL fine-tuning (GRPO, RLVR, R1)

### 8.1
- **edgequake**:
  ```
  "How does GRPO differ from PPO in DeepSeek-R1?"
  "What is RLVR and when is it applicable?"
  ```
- **graphify**:
  ```bash
  graphify path "GRPO" "DeepSeek-R1"
  graphify explain "RLVR"
  ```

---

## 9. Long-context fine-tuning

### 9.1
- **edgequake**:
  ```
  "How to extend context length of pre-trained models for fine-tuning?"
  ```
- **graphify**:
  ```bash
  graphify explain "Long Context Extension Methods"
  graphify path "YaRN" "LongRoPE"
  graphify path "LongLoRA" "Position Interpolation"
  ```

---

## 10. Surprising / cross-community queries (impactantes em apresentação)

Pegando das `Surprising Connections` do GRAPH_REPORT.md:

### 10.1
- **graphify path**:
  ```bash
  graphify path "Block Expansion" "I-LoRA"
  graphify path "Prefix-Tuning" "LoRA"
  graphify path "Evol-Instruct" "Self-Instruct"
  ```

### 10.2 Bridge node insight
- **graphify query**:
  ```bash
  graphify query "Why does Medical Fine-Tuning Long-Context Study connect Instruction Tuning to Continual Learning"
  ```

---

## 11. Code-side (depois que os 1058 .py terminarem de ingerir)

### 11.1 Implementações reais
- **edgequake**:
  ```
  "How is LoRA implemented in HuggingFace PEFT library?"
  "What does the TRL DPOTrainer do?"
  "How does bitsandbytes implement 8-bit quantization?"
  ```

### 11.2 Code-graph navigation (graphify AST)
- **graphify path**:
  ```bash
  graphify path "DPOTrainer" "Trainer"
  graphify path "LoraConfig" "PeftConfig"
  graphify path "Linear8bitLt" "Linear"
  ```

### 11.3 Funções chave
- **graphify explain**:
  ```bash
  graphify explain "DPOTrainer"
  graphify explain "LoraConfig"
  graphify explain "Linear8bitLt"
  graphify explain "compute_loss"
  ```

---

## 12. Para a apresentação (sequência sugerida 15 min)

1. **Setup + problema** (2 min)
2. **Visão de alto nível** — mostra o `graph.html` no browser, comunidades coloridas, god nodes (LoRA central)
3. **Query principal** (3 min):
   ```bash
   graphify query "When to use LoRA vs QLoRA vs full fine-tuning for low-GPU devices"
   ```
   Mostra o subgrafo, traversal BFS, citações com source_url do arxiv
4. **Aprofundamento por eixo** (5 min) — uma query de cada eixo (VRAM/qualidade/speed)
5. **Surprising connection** (1 min) — coloca o slide do "Why does Medical FT connect Instruction Tuning to Continual Learning"
6. **Code grounding** (2 min) — mostra `graphify explain "DPOTrainer"` puxando da peft/trl real
7. **Insight final + limitações** (2 min):
   - Insight: pra <24GB, QLoRA + NF4 + Double Quantization é o sweet spot. DoRA dá +3-5% acc sobre LoRA com mesma VRAM.
   - Limitação: graphify INFERRED edges têm ruído (61% inferred no dataset combinado); edgequake KG depende da qualidade do LLM.

---

## 13. Backup (caso tempo sobrar ou perguntas inesperadas)

```bash
graphify query "Distillation vs PEFT for compressing fine-tuned models" --budget 1500
graphify query "Why does on-policy distillation outperform off-policy" --dfs
graphify path "MiniLLM" "GKD"
graphify explain "Reverse KL Divergence for Distillation"
```

---

**Endpoints úteis durante a demo:**
- WebUI edgequake: http://192.168.120.94:3000  (login: default_user / vitor)
- API edgequake: http://192.168.120.94:8080
- Graphify HTML: `file:///C:/Users/vitor/work/edgequake/dataset-finetuning-research/graphify-out/graph.html`
- Workspace ID: `86f04f70-6f1b-4dea-9db2-f1a54cf95b93` (vllm + .py)
