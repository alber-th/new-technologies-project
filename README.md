# O que faz um jogo ser bem avaliado na Steam?

Trabalho da disciplina **Análise de Dados e Solução de Problemas com Python**
— 1º Semestre de 2026.

**Integrantes:** Alberth Cavalcanti, Ivo Pinheiro.

Projeto de análise exploratória sobre reviews da Steam, com foco em
descobrir **quais características dos jogos** (gênero, tags, faixa de
preço, ano de lançamento, popularidade) estão associadas a uma maior
**taxa de recomendação**, incluindo a busca por combinações não óbvias
("cerveja/fralda"). Entrega uma pipeline reproduzível em Python +
interface interativa em Streamlit + relatório em PDF.

---

## Sumário

1. [Introdução ao Problema](#introdução-ao-problema)
2. [Tecnologias utilizadas](#tecnologias-utilizadas)
3. [Estrutura do projeto](#estrutura-do-projeto)
4. [Pré-requisitos](#pré-requisitos)
5. [Setup do ambiente](#setup-do-ambiente)
6. [Obtenção do dataset](#obtenção-do-dataset)
7. [Execução passo a passo](#execução-passo-a-passo)
8. [Interface Streamlit](#interface-streamlit)
9. [Saídas geradas pelos pipelines](#saídas-geradas-pelos-pipelines)
10. [Solução de problemas comuns](#solução-de-problemas-comuns)
11. [Documentação adicional](#documentação-adicional)
12. [Limitações conhecidas](#limitações-conhecidas)

---

## Introdução ao Problema

A Steam concentra um volume massivo de avaliações de jogadores, em que
cada review é classificada como **Recomendada** ou **Não Recomendada**.
Esse sinal é hoje uma das principais referências de qualidade percebida
de um jogo — mas não é trivial entender, de forma quantitativa, **o que
faz um jogo ser bem avaliado**.

Este projeto investiga como **metadados** dos jogos — gênero, tags, ano
de lançamento, faixa de preço, estúdio/publisher, classificação
indicativa e popularidade — se relacionam com a **taxa de recomendação**.
Mais do que olhar cada dimensão isoladamente, o objetivo é encontrar
**combinações não óbvias** de características (no espírito do clássico
"cerveja e fralda" do varejo) que se associam a jogos consistentemente
bem avaliados, gerando insumo acionável para estúdios, publishers e
analistas de mercado.

Por isso a abordagem usa exclusivamente metadados estruturados — sem
análise de sentimento sobre o texto livre das reviews — e expõe três
camadas de análise: **univariada** (por dimensão), **combinatória**
(pares de características) e **preditiva** (modelo baseline).

---

## Tecnologias utilizadas

| Tecnologia | Uso no projeto |
|---|---|
| **Pandas** | Leitura, limpeza, agregação, mineração de combinações |
| **Numpy** | Categorizações (`np.select` para faixas de preço e ano) |
| **Matplotlib** | Gráficos univariados e de combinações |
| **Scikit-learn** | Pipeline preditivo (`LogisticRegression` + `StandardScaler` + `SimpleImputer`) |
| **Streamlit** | Interface interativa com filtros e visualizações |

Versões fixadas em [`requirements.txt`](./requirements.txt).

---

## Estrutura do projeto

```
new-technologies-project/
├── app_streamlit.py              # interface interativa (Streamlit)
├── main_diagnostico.py           # 1: sanity check do CSV bruto
├── main_preprocessing.py         # 2: limpeza + feature engineering
├── main_eda.py                   # 3: análise univariada
├── main_patterns.py              # 4: mineração de combinações
├── main_modeling.py              # 5: modelo baseline (opcional)
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loading.py           # leitura do CSV bruto + diagnóstico
│   ├── preprocessing.py          # clean_raw_data + engineer_features
│   ├── eda.py                    # plots univariados (devolvem Figure)
│   ├── patterns.py               # combinações + top combos + Venn-like
│   ├── modeling.py               # pipeline ML baseline
│   └── viz_utils.py              # (placeholder)
├── data/
│   ├── raw/                      # CSV bruto (NÃO versionado)
│   └── processed/                # CSV pós-pipeline (NÃO versionado)
└── docs/
    ├── relatorio.md              # relatório do projeto em Markdown
    ├── figures/                  # PNGs gerados pelos scripts main_*
    ├── Ideia base do Projeto.md
    ├── Instruções do Projeto..md
    ├── PlanejamentoClaudeCodeNovasTecnologias.md
    └── Pontos a se Analisar.md
```

---

## Pré-requisitos

- **Python 3.10 ou superior** (o código usa sintaxe de tipos moderna
  como `set[str]`, `Series | None`).
- **Git** (para clonar o repositório).
- **Conta no Kaggle** ou uso do `kaggle` CLI (para baixar o dataset).
- ~500 MB de espaço em disco para o dataset e artefatos gerados.

---

## Setup do ambiente

```bash
# 1. Clonar o repositório
git clone https://github.com/alber-th/new-technologies-project.git
cd new-technologies-project

# 2. Criar e ativar virtual environment
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Obtenção do dataset

**Dataset escolhido:** [Steam Game Reviews](https://www.kaggle.com/datasets/smeeeow/steam-game-reviews)
(`smeeeow/steam-game-reviews` no Kaggle).

**Critério de escolha:** combina o campo binário de recomendação
(`recommended`) com metadados ricos (gênero, tags, data de lançamento,
desenvolvedor/publisher, preço), o que permite responder à pergunta de
negócio com foco em metadados — sem precisar de NLP pesado sobre o
texto das reviews.

### Opção A — Download manual (mais simples)

1. Acesse a página do dataset no Kaggle.
2. Faça login e clique em **Download**.
3. Extraia o arquivo `.zip`.
4. Renomeie o CSV principal para **`steam_reviews.csv`** e mova para a
   pasta `data/raw/` do projeto.

Caminho final esperado:

```
data/raw/steam_reviews.csv
```

### Opção B — Via Kaggle CLI

```bash
# Pré-requisito: kaggle CLI instalado e API key configurada
# (https://github.com/Kaggle/kaggle-api#api-credentials)

pip install kaggle
mkdir -p data/raw
kaggle datasets download -d smeeeow/steam-game-reviews -p data/raw/ --unzip

# Se o CSV extraído tiver outro nome, renomeie:
mv data/raw/<nome_real>.csv data/raw/steam_reviews.csv
```

### Importante: nomes de colunas

O código em `src/preprocessing.py` assume nomes de coluna padronizados
(`recommended`, `app_id`, `app_name`, `tags`, `genres`, `release_date`,
`price`, etc.). Esses nomes estão centralizados em **constantes
`COL_*` no topo do arquivo**.

Se o dataset baixado usar nomenclatura diferente (ex.: `is_recommended`
em vez de `recommended`), basta editar essas constantes — os módulos
ignoram silenciosamente colunas que não existem, então o pipeline não
quebra.

**O script `main_diagnostico.py` (passo 1 da execução) mostra
exatamente quais colunas estão no seu CSV.** Rode-o primeiro para
verificar se algum ajuste é necessário.

---

## Execução passo a passo

A pipeline foi desenhada para rodar em sequência, cada `main_*.py`
depende da saída do anterior (do passo 2 em diante).

### Passo 1 — Diagnóstico do CSV bruto

```bash
python main_diagnostico.py
```

**O que faz:** carrega `data/raw/steam_reviews.csv` e imprime no
console: dimensões, tipos de dados, contagem de nulos por coluna e
amostra de 10 linhas. Útil para conferir se os nomes de coluna batem
com as constantes `COL_*` antes de rodar o pipeline pesado.

**Quando ajustar:** se alguma coluna esperada (`recommended`, `tags`,
etc.) não aparece, edite as constantes em `src/preprocessing.py` para
o nome real.

### Passo 2 — Limpeza e feature engineering

```bash
python main_preprocessing.py
```

**O que faz:**
- Deduplica reviews (por `review_id` ou fallback combinado).
- Converte datas para `datetime`.
- Binariza a recomendação (`1` = recomendado, `0` = não).
- Descarta linhas sem alvo ou sem `app_id`.
- Cria features derivadas: `price_band`, `release_year_band`, flags
  `has_action`/`has_indie`/`has_rpg`/`has_coop`/`has_multiplayer`/
  `has_singleplayer`, `num_reviews_game`.

**Saída:** `data/processed/steam_reviews_processed.csv` (base para os
próximos passos) + resumo das novas colunas no stdout.

### Passo 3 — Análise exploratória (EDA)

```bash
python main_eda.py
```

**O que faz:** gera quatro gráficos univariados (taxa de recomendação
por gênero, faixa de preço, período de lançamento e top tags por nº de
reviews) e imprime estatísticas resumo (taxa global, top 5 flags por
taxa, melhor/pior faixa de preço e período).

**Saídas em `docs/figures/`:**
- `rec_rate_by_genre.png`
- `rec_rate_by_price.png`
- `rec_rate_by_period.png`
- `top_genres_by_review_count.png`

### Passo 4 — Mineração de padrões

```bash
python main_patterns.py
```

**O que faz:** ranqueia as top 10 combinações de duas características
com maior taxa de recomendação (filtrando combinações com menos de
50 jogos únicos), calcula **lift** (razão sobre o baseline) e gera a
visualização "Venn-like" para pares de flags selecionados.

**Saídas em `docs/figures/`:**
- `top_combinations.png`
- `overlap_indie_coop.png`
- `overlap_indie_multiplayer.png`
- `overlap_action_rpg.png`

Tabela com as top combinações é impressa no console.

### Passo 5 — Modelo baseline (opcional)

```bash
python main_modeling.py
```

**O que faz:** treina um pipeline `SimpleImputer → StandardScaler →
LogisticRegression(class_weight="balanced")` em 80% dos dados (split
estratificado pelo alvo), avalia no 20% restante e imprime accuracy,
precision, recall, F1, matriz de confusão e as top 10 features
associadas a maior probabilidade de recomendação.

### Passo 6 — Interface Streamlit

```bash
streamlit run app_streamlit.py
```

Abre automaticamente em [http://localhost:8501](http://localhost:8501).
Detalhes na próxima seção.

---

## Interface Streamlit

A aplicação consolida toda a análise em cinco seções:

1. **Introdução** — contexto curto do projeto.
2. **Visão geral** — KPIs do recorte atual (reviews, jogos únicos, taxa
   média de recomendação, gênero mais frequente).
3. **Análise por dimensão** — abas com gráficos univariados.
4. **Padrões escondidos** — top combinações + visualização "Venn-like"
   interativa.
5. **Modelo baseline** *(opcional, habilitado por toggle na sidebar)* —
   métricas, top features e formulário de predição interativa.

A **sidebar** oferece filtros por:
- Gêneros/modos de jogo (flags `has_*`, união);
- Faixa de preço (Grátis / Barato / Médio / Caro);
- Período de lançamento (antes de 2010 até depois de 2020).

Os filtros se aplicam a todas as seções (exceto à do modelo, que é
treinado uma única vez sobre o dataset completo, com cache).

**Pré-requisito:** ter rodado `main_preprocessing.py` antes — o app lê
direto de `data/processed/steam_reviews_processed.csv`.

---

## Saídas geradas pelos pipelines

Após rodar a sequência completa, esses artefatos existirão:

```
data/processed/
└── steam_reviews_processed.csv     # ~MB, depende do dataset

docs/figures/
├── rec_rate_by_genre.png
├── rec_rate_by_price.png
├── rec_rate_by_period.png
├── top_genres_by_review_count.png
├── top_combinations.png
├── overlap_indie_coop.png
├── overlap_indie_multiplayer.png
└── overlap_action_rpg.png
```

Esses PNGs podem ser anexados ao relatório (ver `docs/relatorio.md`)
para a versão final em PDF.

---

## Solução de problemas comuns

### `FileNotFoundError: data/raw/steam_reviews.csv`

Você não baixou o dataset ou ele está com outro nome. Ver
[Obtenção do dataset](#obtenção-do-dataset).

### `KeyError` em uma coluna esperada

O dataset baixado usa nomes diferentes dos esperados pelas constantes
`COL_*` em `src/preprocessing.py`. Rode `python main_diagnostico.py`
para listar as colunas reais e ajuste as constantes correspondentes.

### `ModuleNotFoundError: No module named 'src'`

Você está rodando o script de fora da raiz do projeto. Sempre execute
os scripts a partir do diretório `new-technologies-project/`:

```bash
cd new-technologies-project
python main_preprocessing.py        # ✓
```

### Gráficos não aparecem no Streamlit / muito lentos

Os caches do Streamlit ficam guardados em memória durante a sessão. Se
você alterou o CSV processado, force o refresh pelo menu "..." do
Streamlit ou reinicie o processo (`Ctrl+C` e `streamlit run` de novo).

### Modelo demora muito para treinar

Acontece apenas na **primeira vez** que a seção do modelo é aberta no
Streamlit (`@st.cache_resource` mantém o modelo em memória depois). Se
quiser pular o modelo, deixe o toggle "Mostrar seção do modelo"
desativado.

---

## Documentação adicional

- [`docs/relatorio.md`](./docs/relatorio.md) — relatório textual do
  projeto em Markdown (pronto para exportar como PDF).
- [`docs/Ideia base do Projeto.md`](./docs/Ideia%20base%20do%20Projeto.md)
  — definição inicial.
- [`docs/Pontos a se Analisar.md`](./docs/Pontos%20a%20se%20Analisar.md)
  — pontos de análise levantados no escopo.
- [`docs/figures/`](./docs/figures/) — figuras geradas pelos pipelines
  (criadas após executar os scripts `main_*.py`).

---

## Limitações conhecidas

- O dataset usa apenas metadados publicamente disponíveis; dados
  detalhados de jogadores (faixa etária, país, horas jogadas), preço
  regional histórico e pico de jogadores simultâneos podem não estar
  representados.
- O parser de tags depende de strings padronizadas; tags raras ou em
  outros idiomas podem ter sido subcontadas.
- O modelo de ML é um **baseline interpretável** (regressão logística
  linear). Não captura interações sem feature engineering adicional —
  por isso o módulo `patterns.py` complementa essa limitação fazendo a
  mineração explícita de combinações.
- Reviews são tratadas como observações independentes — jogos com
  muitos reviews ponderam mais nas agregações por categoria.

Ver discussão completa em [`docs/relatorio.md`](./docs/relatorio.md).
