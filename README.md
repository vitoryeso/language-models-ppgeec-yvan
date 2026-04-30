# Modelos de Linguagem — PPgEEC UFRN

Trabalhos da disciplina **Modelos de Linguagem** do Programa de Pós-Graduação em Engenharia Elétrica e de Computação (PPgEEC) — Universidade Federal do Rio Grande do Norte (UFRN).

Aluno: **Vitor Y. F. Freitas** · `vitoryeso@gmail.com`

---

## Trabalho 01 — Fundamentos

Materiais introdutórios da disciplina:
- Transcrição dos vídeos do curso *Intro to Large Language Models* do Andrej Karpathy
- Diagrama-completo do pipeline de pré-treino → SFT → RLHF (Excalidraw + SVG)
- Atividade-1 zipada
- Ícones e materiais de apresentação

→ [`trabalho-01/`](trabalho-01/)

## Trabalho 02 — Estudo de Caso: Knowledge Graphs para Fine-Tuning

**Pergunta:** *quando usar LoRA, QLoRA ou Full Fine-Tuning sob restrição de VRAM?*

**Abordagem:** construir um knowledge graph **incrementalmente** (papers → docs de bibliotecas → código-fonte) e responder a pergunta com **rastreabilidade às fontes**, mostrando como a profundidade das respostas evolui com o aterramento do corpus. Apresentado em **2026-04-30**.

**Equipe:** Vitor Y. F. Freitas · Reilta · Luis Henrique.

→ [`trabalho-02/`](trabalho-02/) · [`trabalho-02/README.md`](trabalho-02/README.md) · [`trabalho-02/stages_comparison_synthesized.md`](trabalho-02/stages_comparison_synthesized.md)

---

## Estrutura

```
.
├── README.md           (este arquivo — overview da disciplina)
├── trabalho-01/        (fundamentos + Karpathy transcripts + diagramas)
└── trabalho-02/        (case study fine-tuning — graphify + edgequake)
```
