# Trabalho 02 — Knowledge Graphs para Decisões de Fine-Tuning de LLMs

**Disciplina:** Modelos de Linguagem · PPgEEC UFRN · 2026
**Tema do trabalho:** Fine-Tuning / Training
**Apresentação:** 2026-04-30
**Equipe:** Vitor Y. F. Freitas · Reilta · Luis Henrique

---

## Pergunta de pesquisa

> *Dado um cenário de adaptação supervisionada de LLMs em hardware com VRAM ≤ 24 GB, **quando usar LoRA, QLoRA ou Full Fine-Tuning?***

Eixos comparativos: consumo de VRAM · custo computacional · qualidade no downstream · requisitos de hardware · esquecimento catastrófico.

## Tese

Construir um **knowledge graph navegável** sobre o domínio de fine-tuning, **incrementalmente** (papers → docs de bibliotecas → código-fonte), e mostrar que **a profundidade das respostas a uma pergunta cresce monotonicamente com o aterramento do corpus**, ainda que a **decisão final seja estável** já no Stage 1.

O foco do trabalho não é fazer fine-tuning real, mas **extrair e organizar conhecimento** suficiente pra responder a pergunta com **rastreabilidade às fontes primárias**.

## Ferramentas utilizadas

| Ferramenta | Papel | Status |
|---|---|---|
| **graphify** | Extração de KG (AST + sub-agents Claude), clustering Louvain, viz HTML/SVG | ✅ rodado nos 3 stages |
| **edgequake** | RAG sobre KG (chunk + embed + KG extraction via LLM, pgvector) | ⚠️ parcial — interrompido por reboot |
| **vLLM** (Qwen3-14B FP8) + **TEI** (e5-large 1024-d) em cluster 2× RTX 4090 | LLM + embeddings pro edgequake | usado durante Stage 3 |
| **claude-sonnet** via `claude -p` + skill graphify | Síntese das respostas a partir do subgrafo BFS | 7 queries × 3 stages = 21 respostas |

A apresentação focou em graphify — edgequake teve problemas de infra (turing reboot mid-pipeline, GPU disputada com outro projeto). A comparação 3-stage no edgequake fica como trabalho futuro.

## Construção incremental

| Stage | Corpus adicionado | Total arquivos | Nodes | Edges | Comunidades |
|---|---|---|---|---|---|
| **1** | 54 papers (arXiv 2024–2026 + foundational) | 54 | 118 | 168 | 11 |
| **2** | + 77 docs de `peft/trl/bitsandbytes` (`docs/source/`) | 131 | 294 | 368 | 18 |
| **3** | + 1058 `.py` de `pytorch/transformers/peft/trl/bitsandbytes` (AST) | 1189 | **18 803** | **66 935** | **288** |

**Inversão dos god nodes** entre stages é o highlight visual:

| Stage | Top god node | Tipo |
|---|---|---|
| 1 | `LoRA: Low-Rank Adaptation` | conceito de paper |
| 2 | `LoraConfig` | classe Python (peft API) |
| 3 | `LoraLayer` / `LoraParallelLinear` | implementação (módulo neural) |

## Estrutura do repositório

```
trabalho-02/
├── README.md                              ← este arquivo
├── dataset-links/                         ← 54 stubs com YAML frontmatter (~50KB total)
│   ├── manifest.json                      ← índice com title/source_url/topics/date
│   └── 01-sft/, 02-lora/, ..., 10-continual/
├── stages/
│   ├── 1/                                 papers só
│   │   ├── graph.json                     ← grafo serializado (NetworkX node-link)
│   │   ├── graph.html                     ← viz interativa (vis.js)
│   │   ├── GRAPH_REPORT.md                ← god nodes + comunidades + surprising connections
│   │   └── answers/q01.md … q07.md        ← 7 respostas sintetizadas
│   ├── 2/                                 papers + lib docs
│   └── 3/                                 papers + lib docs + código (35 MB graph.json)
├── graphify-viz/                          ← visualizações compactas do grafo Stage 3
│   ├── graph_papers.html                  ← 118 nodes (papers só) — 100 KB
│   ├── graph_communities.html             ← 288 super-nodes (uma por comunidade) — 157 KB
│   └── graph_top500.html                  ← top-500 por grau — 798 KB
├── stages_comparison_synthesized.md       ← 21 respostas em formato Pergunta/Modelo/Resposta
├── queries.md                             ← catálogo de 60+ queries planejadas
└── tasks.txt                              ← patch pendente: relaxar `is_model_provider_mismatch` no edgequake
```

## Quick navigation

- **Quer ler as respostas finais comparadas?** → [`stages_comparison_synthesized.md`](stages_comparison_synthesized.md)
- **Quer ver o grafo no browser?** → abra `graphify-viz/graph_papers.html` ou `graphify-viz/graph_communities.html` localmente
- **Quer entender a estrutura/comunidades de cada stage?** → [`stages/1/GRAPH_REPORT.md`](stages/1/GRAPH_REPORT.md), [`stages/2/`](stages/2/GRAPH_REPORT.md), [`stages/3/`](stages/3/GRAPH_REPORT.md)
- **Quer reproduzir uma resposta?** → ver [Reprodução](#reprodução)

## Reprodução

### 1. Recriar o corpus

`dataset-links/manifest.json` lista os 54 papers com `source_url`. Para baixar o conteúdo pleno:

```bash
pip install graphifyy
mkdir corpus && cd corpus
for url in $(jq -r '.files[].source_url' ../dataset-links/manifest.json); do
  graphify add "$url" --dir ./raw
done
```

Para clonar as bibliotecas usadas na Stage 3 (~80 MB shallow):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/pytorch/pytorch.git
git clone --depth 1 https://github.com/huggingface/transformers.git
git clone --depth 1 https://github.com/huggingface/peft.git
git clone --depth 1 https://github.com/huggingface/trl.git
git clone --depth 1 https://github.com/bitsandbytes-foundation/bitsandbytes.git
```

### 2. Reconstruir o grafo

A skill `/graphify <path>` no Claude Code dispara o pipeline completo (detect → AST + extração semântica via sub-agents → cluster Louvain → relatório).

```bash
# em Claude Code, num diretório com o corpus
/graphify .
```

Ou, via CLI direta (apenas AST, sem extração semântica):
```bash
graphify update .
```

### 3. Rodar uma query sintetizada

Cada `stages/N/graph.json` é autocontido:

```bash
cd stages/2
mkdir -p graphify-out
cp graph.json graphify-out/
claude -p --model sonnet --dangerously-skip-permissions \
  "Use the graphify skill (graph at graphify-out/graph.json) to query: 'sua pergunta'"
```

## Achados principais

1. **A decisão prática é estável já em Stage 1.** Para qualquer cenário de VRAM ≤ 24 GB, **QLoRA é o default**. Adicionar docs de biblioteca ou código não muda a recomendação de método — só aumenta a especificidade da prescrição.
2. **A profundidade da prescrição cresce com o stage:**
   - Stage 1 → *"use QLoRA"*
   - Stage 2 → *"instancie `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')` e use com `LoraConfig(r=16, lora_alpha=32)`"*
   - Stage 3 → *"`Linear4bit` herda de `nn.Linear`, integra com `prepare_model_for_kbit_training`, paged optim em `bitsandbytes/optim/optimizer.py`"*
3. **Stage 2 é o sweet spot pra um estudo de caso prático** — mantém rastreabilidade às fontes primárias (papers) sem o ruído estrutural que 18.000 nodes do Stage 3 introduzem.
4. **Surprising connection** detectada por betweenness cross-community: o nó *Medical Fine-Tuning Long-Context Study* faz ponte entre "Instruction Tuning & Long Context" e "Continual Learning & Forgetting" — sinaliza que escolha de método pode ser determinada por restrições de domínio (forgetting crítico ⇒ LoRA isolado prevalece sobre QLoRA mesmo com VRAM).

## Limitações honestas

- **61% das edges em Stage 3 são INFERRED** (avg confidence 0.68), vs 32% em Stage 1 — a síntese depende da qualidade do extrator semântico (Claude Sonnet via skill).
- **Validação experimental ausente** — nenhum fine-tuning real foi executado para corroborar quantitativamente as recomendações. O escopo do trabalho é análise estruturada, não experimentação.
- **Edgequake comparison incompleta** — pipeline interrompido por reboot do nó de cálculo (turing) durante a Stage 3, e GPU subsequentemente alocada a outro projeto. O grafo do Stage 3 ainda foi construído via graphify (AST), mas o RAG-via-edgequake ficou parcial; comparação cross-stage no edgequake fica como trabalho futuro.
- **GraphML export falha** (limitação de `networkx.write_graphml_lxml` com atributos `list`); HTML, SVG e JSON são suficientes pros casos de uso aqui.

## Stack técnica

`graphify` (Karpathy `/raw` workflow) · `edgequake` (Rust) · `vLLM` (Qwen3-14B FP8) · `TEI` (e5-large 1024-d) · `claude-sonnet` via skill · `pgvector` · `Apache AGE` · `nginx` LB · `Tailscale Funnel`.

## Contato

**Vitor Y. F. Freitas** · `vitoryeso@gmail.com` · PPgEEC UFRN
