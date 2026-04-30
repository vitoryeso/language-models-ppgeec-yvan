# Trabalho 02 — Estudo de Caso: Knowledge Graphs para Decisões de Fine-Tuning de LLMs

**Disciplina:** Modelos de Linguagem · PPgEEC UFRN · 2026
**Tema:** Fine-Tuning / Training
**Apresentado:** 2026-04-30

**Pergunta de pesquisa:** dado um cenário de adaptação supervisionada de LLMs em hardware com VRAM ≤ 24 GB, quando usar **LoRA**, **QLoRA** ou **Full Fine-Tuning**?

## Resumo

Construímos um **knowledge graph navegável** sobre o domínio de fine-tuning de LLMs e usamos o grafo para responder, com rastreabilidade às fontes, perguntas comparativas entre métodos. O trabalho não envolve fine-tuning real — o foco é na **extração estruturada de conhecimento** a partir de papers, documentação e código.

A construção é **incremental em três stages**, mostrando como a profundidade das respostas evolui com o nível de aterramento do corpus:

| Stage | Corpus | Nodes | Edges | Comunidades |
|---|---|---|---|---|
| **1** | 54 papers (arXiv 2024–2026 + foundational) | 118 | 168 | 11 |
| **2** | + 77 docs `peft/trl/bitsandbytes` (`docs/source/`) | 294 | 368 | 18 |
| **3** | + 1058 `.py` de `pytorch/transformers/peft/trl/bitsandbytes` (AST) | 18 803 | 66 935 | 288 |

A mesma pergunta foi feita aos 3 grafos. A resposta evolui de **conceitual → API → linha de código** (ver `stages_comparison_synthesized.md`).

## Ferramentas utilizadas

- **graphify** — extração de KG (AST + LLM), clustering Louvain, viz HTML/SVG
- **edgequake** — RAG sobre KG (chunk + embed + KG extraction via LLM, pgvector)
- **vLLM** (Qwen3-14B FP8) + **TEI** (e5-large) em cluster de RTX 4090
- **claude-sonnet** via `claude -p` para síntese das respostas usando a skill graphify

## Estrutura do repositório

```
trabalho-02/
├── README.md                            (este arquivo)
├── dataset-links/                       (54 stubs com YAML frontmatter; fetch via source_url)
│   ├── manifest.json
│   └── 01-sft/, 02-lora/, ..., 10-continual/
├── stages/
│   ├── 1/  (graph.json, GRAPH_REPORT.md, graph.html, answers/q01..q07.md)
│   ├── 2/
│   └── 3/
├── graphify-viz/
│   ├── graph_papers.html                (118 nodes — leve)
│   ├── graph_communities.html           (288 super-nodes — visão macro)
│   └── graph_top500.html                (top-degree do grafo full)
├── stages_comparison_synthesized.md     (3-stage Q&A consolidado)
├── queries.md                           (60+ queries graphify/edgequake)
└── tasks.txt                            (tarefas/patches pendentes)
```

## Como reproduzir

### 1. Recriar o corpus
Os arquivos do `dataset-links/` contêm apenas `source_url` no frontmatter. Para baixar o conteúdo:

```bash
# exemplo: usar graphify para baixar os arquivos
for url in $(jq -r '.files[].source_url' dataset-links/manifest.json); do
  graphify add "$url"
done
```

Para clonar as bibliotecas usadas na Stage 3:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/pytorch/pytorch.git
git clone --depth 1 https://github.com/huggingface/transformers.git
git clone --depth 1 https://github.com/huggingface/peft.git
git clone --depth 1 https://github.com/huggingface/trl.git
git clone --depth 1 https://github.com/bitsandbytes-foundation/bitsandbytes.git
```

### 2. Reproduzir o grafo
```bash
pip install graphifyy
cd <corpus_dir>
# Stage 1: papers só
graphify update .   # AST-only — para Stage 1 usamos a skill no Claude Code
# Stage 2 e 3: usar a skill /graphify <path> em Claude Code (despacha sub-agents)
```

### 3. Reproduzir as respostas sintetizadas
Cada `stages/N/graphify-out/graph.json` é independente. Para rodar uma query:
```bash
cd stages/2
claude -p --model sonnet --dangerously-skip-permissions \
  "Use the graphify skill (graph at graphify-out/graph.json) to query: '<sua pergunta>'"
```

## Achados principais

1. **Conclusão de decisão é estável atravessando stages**: QLoRA é default abaixo de 24 GB VRAM em todas as 3 versões do grafo.
2. **Profundidade da prescrição cresce**: Stage 1 → "use QLoRA"; Stage 2 → "instancie `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')`"; Stage 3 → linha exata do `Linear4bit` no bitsandbytes.
3. **Inversão dos god nodes** entre stages: paper LoRA → classe `LoraConfig` → camada `LoraLayer`. Mostra que o eixo conceitual do domínio se desloca conforme o aterramento do corpus.

## Limitações honestas

- 61% das edges em Stage 3 são INFERRED (vs 32% em Stage 1) — síntese depende da qualidade do extrator semântico.
- Validação experimental ausente — nenhum fine-tuning real foi executado.
- Edgequake (RAG sobre KG, comparação por stage) ficou parcial: pipeline interrompido por reboot do nó de cálculo durante a Stage 3.

## Equipe

- **Vitor Y. F. Freitas** — `vitoryeso@gmail.com` · PPgEEC UFRN
- **Reilta** (PPgEEC UFRN)
- **Luis Henrique** (PPgEEC UFRN)

## Stack técnica

`graphify` · `edgequake` · `vLLM` (Qwen3-14B FP8) · `TEI` (e5-large 1024-d) · `claude-sonnet` (skill graphify) · `pgvector` · `Apache AGE` · `nginx` LB · `Tailscale Funnel`.
