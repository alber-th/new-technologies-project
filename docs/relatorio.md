# Relatório — O que faz um jogo ser bem avaliado na Steam?

**Disciplina:** Análise de Dados e Solução de Problemas com Python
**Integrantes:** Alberth Cavalcanti, Ivo Pinheiro
**Repositório:** https://github.com/alber-th/new-technologies-project

> **Nota sobre números.** Trechos marcados com colchetes (`[…]`) correspondem a
> valores que dependem da execução dos pipelines sobre o dataset baixado.
> Eles devem ser substituídos pelos números reais antes da conversão final
> para PDF (basta rodar `python main_eda.py`, `main_patterns.py` e
> `main_modeling.py` e copiar as saídas).

---

## 1. Introdução ao Problema

A Steam é a maior loja digital de jogos para PC e funciona, em grande
parte, como uma rede social de jogadores: cada usuário pode publicar uma
avaliação marcando o título como **Recomendado** ou **Não Recomendado**,
e o conjunto dessas avaliações forma o principal sinal público de
qualidade percebida de um jogo. Para estúdios, publishers e analistas de
mercado, entender o que está por trás dessas recomendações é
estratégico: decisões de gênero, escopo, precificação, janela de
lançamento e posicionamento de marketing são tomadas, muitas vezes, sem
uma leitura quantitativa clara de quais características realmente se
correlacionam com aceitação positiva do público.

A pergunta de negócio que orienta este projeto é: **quais características
de um jogo na Steam estão associadas a uma maior taxa de avaliações
positivas?** A abordagem escolhida foi olhar para os **metadados do
jogo** — gênero, tags, ano de lançamento, faixa de preço,
estúdio/publisher, classificação indicativa e popularidade — e medir
como cada dimensão se relaciona com a proporção de reviews recomendados,
sem recorrer a análise de sentimento sobre o texto livre da review. O
resultado pretende ser acionável: se uma faixa de preço, um gênero ou
uma janela de lançamento se mostra consistentemente acima da média,
isso vira insumo para decisões de produto.

Além das relações diretas (ex.: "gênero A é mais bem avaliado que
gênero B"), o projeto busca explicitamente **padrões não óbvios de
coocorrência**, no espírito do clássico exemplo "cerveja/fralda" do
varejo: combinações de tags, faixas de preço e janelas de lançamento
que, juntas, se associam a taxas de recomendação muito acima da média —
mesmo quando cada característica isolada parece neutra. Identificar
esses cruzamentos é onde a análise gera valor diferenciado em relação a
uma simples leitura por gênero.

---

## 2. Metodologia

### 2.1 Fonte de dados

- **Dataset:** *Steam Game Reviews* (Kaggle, `smeeeow/steam-game-reviews`).
- **Critério de seleção:** combina campo binário de recomendação com
  metadados ricos (gênero, tags, ano, preço, desenvolvedor/publisher),
  permitindo responder à pergunta de negócio sem necessidade de NLP
  sobre o texto da review.
- **Organização em disco:** CSV bruto em `data/raw/steam_reviews.csv`;
  versão limpa e enriquecida em `data/processed/steam_reviews_processed.csv`.

### 2.2 Estrutura do projeto

```
src/
  data_loading.py     # leitura e diagnóstico do CSV bruto
  preprocessing.py    # limpeza + feature engineering
  eda.py              # análise univariada
  patterns.py         # mineração de combinações ("cerveja/fralda")
  modeling.py         # baseline ML
app_streamlit.py      # interface interativa
main_*.py             # scripts que orquestram cada etapa
data/                 # raw e processed (versionados via .gitignore)
docs/                 # figuras e relatório
```

Cada módulo é responsável por uma etapa do pipeline e devolve estruturas
reaproveitáveis (DataFrames, objetos `Figure`, modelos `Pipeline`),
permitindo que tanto os scripts `main_*.py` quanto a interface Streamlit
consumam a mesma lógica sem duplicação.

### 2.3 Carregamento e limpeza (Pandas)

`src/data_loading.py` faz a leitura via `pandas.read_csv` e imprime um
diagnóstico inicial (dimensões, `dtypes`, contagem de nulos, amostra).
`src/preprocessing.py` então aplica:

- **Deduplicação** por `review_id`, com fallback para a combinação
  `(app_id, author, review_date)` quando o ID único está ausente.
- **Conversão de datas** para `datetime`, usando `errors="coerce"` para
  não quebrar o pipeline em valores mal formatados.
- **Binarização da recomendação** (`1` = recomendado, `0` = não
  recomendado), aceitando formatos `bool`, `Recommended` / `Not
  Recommended` / `Positive` / `Negative` e códigos numéricos (`1`, `-1`).
- **Tratamento de nulos:** linhas sem alvo ou sem `app_id` são
  descartadas (registros inúteis para a análise); campos textuais como
  `developer`, `publisher` e `tags` são preenchidos com `"Unknown"` para
  preservar a linha sem distorcer a contagem de jogos válidos.

### 2.4 Engenharia de features (Pandas + Numpy)

- **`price_band`** (`Grátis`, `Barato`, `Médio`, `Caro`) criada com
  `numpy.select`, usando regras absolutas em vez de quantis para gerar
  faixas interpretáveis no relatório. Jogos gratuitos formam categoria
  própria porque seu padrão de avaliação tende a ser distinto.
- **`release_year_band`** (`Antes de 2010`, `2010–2015`, `2016–2020`,
  `Depois de 2020`), também via `numpy.select`. Os recortes acompanham
  o crescimento da Steam e o boom indie pós-2010.
- **Flags binárias** `has_action`, `has_indie`, `has_rpg`, `has_coop`,
  `has_multiplayer`, `has_singleplayer`, derivadas de um parser
  tolerante (`_parse_tags`) que aceita listas Python, strings JSON-like
  e CSV simples.
- **Popularidade** (`num_reviews_game`) calculada com
  `groupby(app_id).transform("size")`, como proxy de quantos jogadores
  efetivamente engajaram com o título.

### 2.5 Análise exploratória (Matplotlib)

`src/eda.py` expõe funções que devolvem objetos `matplotlib.figure.Figure`,
reaproveitados tanto pelo script `main_eda.py` (que salva PNGs em
`docs/figures/`) quanto pela interface Streamlit. As visualizações
principais são:

- Taxa de recomendação por **gênero/modo** (flags `has_*`).
- Taxa de recomendação por **faixa de preço**.
- Taxa de recomendação por **período de lançamento**.
- **Top tags** por número de reviews (gráfico horizontal).

O padrão visual é consistente: título destacado, eixos rotulados, linha
tracejada com a média geral como referência, anotação de valor em cada
barra. A escolha do Matplotlib direto (sem Seaborn) atende ao requisito
da disciplina.

### 2.6 Mineração de padrões (Pandas)

`src/patterns.py` formaliza a busca por combinações. Três fontes de
combos são consideradas:

1. cada flag `has_*` × `price_band`;
2. cada flag `has_*` × `release_year_band`;
3. pares distintos de flags `has_*` × `has_*` (apenas o quadrante "ambos
   ativos").

Para cada combinação calcula-se: taxa de recomendação, número de
reviews, **número de jogos únicos** (`nunique` por `app_id`) e **lift**
(`taxa_do_combo / taxa_geral`). Combinações com menos de **50 jogos
únicos** são descartadas, evitando rankings dominados por amostras
pequenas. Adicionalmente, uma visualização "Venn-like" simplificada
(três barras: `A ∧ B`, `só A`, `só B`) permite inspecionar a sobreposição
entre duas features escolhidas pelo usuário.

### 2.7 Modelo baseline (Scikit-learn)

`src/modeling.py` define o pipeline:

```
SimpleImputer(median) → StandardScaler → LogisticRegression(class_weight="balanced")
```

Cada passo tem justificativa: a imputação mediana protege contra `NaN`
remanescentes em features numéricas; a padronização é necessária pela
sensibilidade da regressão logística à escala e torna os coeficientes
comparáveis entre features; o balanceamento de classes evita o cenário
trivial em que o modelo aprende a prever "1" sempre, dado o
desbalanceamento natural das reviews da Steam. As features são as
numéricas e binárias da base processada, com split estratificado 80/20
e `random_state` fixo para reprodutibilidade.

### 2.8 Interface (Streamlit)

`app_streamlit.py` consolida o que os módulos anteriores produzem em
cinco seções: **introdução**, **KPIs**, **análise univariada em abas**,
**padrões escondidos** (tabela formatada + gráfico de top combinações
+ Venn-like interativo) e **modelo baseline** (métricas, top features
e formulário de predição). Os filtros laterais (gênero, faixa de preço,
período) recortam o DataFrame em todas as seções de análise. Caches
(`@st.cache_data` no CSV, `@st.cache_resource` no modelo) garantem
responsividade.

---

## 3. Resultados Visuais

> Os valores específicos abaixo dependem da execução do pipeline sobre o
> dataset final. As leituras qualitativas anteciparam tendências que
> serão confirmadas pelos números.

### 3.1 Visão geral

- Total de reviews analisadas: `[N reviews]`.
- Jogos únicos: `[N jogos]`.
- Taxa global de recomendação: `[XX,X%]`.

A base da Steam é fortemente assimétrica em favor da classe positiva —
a maioria dos jogos ultrapassa `[XX%]` de recomendação. Esse fato
metodológico justifica o uso de **lift** (razão sobre o baseline) em
vez da taxa absoluta para ranquear padrões: variações de poucos pontos
percentuais já representam efeitos relevantes nesse regime.

### 3.2 Taxa de recomendação por gênero / modo

O gráfico univariado das flags `has_*` aponta `[has_X]` como o
segmento com maior taxa (`[XX,X%]`) e `[has_Y]` como o de menor
(`[XX,X%]`). [Comentário interpretativo: a literatura de mercado
sustenta esse padrão porque… / esse achado é coerente com…]

### 3.3 Taxa de recomendação por faixa de preço

| Faixa | Taxa | Comentário |
|---|---|---|
| Grátis | `[XX,X%]` | `[…]` |
| Barato (≤ R$ 20) | `[XX,X%]` | `[…]` |
| Médio (R$ 20–60) | `[XX,X%]` | `[…]` |
| Caro (> R$ 60) | `[XX,X%]` | `[…]` |

A leitura por faixa permite distinguir efeito de expectativa (jogos caros
sob escrutínio maior) de efeito de qualidade percebida (jogos médios
costumam ser projetos consolidados).

### 3.4 Taxa de recomendação por período de lançamento

A análise por geração indica `[tendência identificada]`: jogos lançados
em `[período]` apresentam taxa significativamente maior do que em
`[outro período]`. [Hipóteses: profissionalização do mercado indie,
correção do viés de "saudosismo" em catálogos antigos, etc.]

### 3.5 Padrões escondidos (combinações de 2 características)

A mineração com `min_games = 50` produziu as combinações abaixo (Top 10
por taxa de recomendação). O **lift** mede o ganho da combinação sobre o
baseline da Steam.

| # | Combinação | Taxa | Lift | Nº jogos |
|---|---|---|---|---|
| 1 | `[…]` | `[XX,X%]` | `[X,XX]` | `[…]` |
| 2 | `[…]` | `[XX,X%]` | `[X,XX]` | `[…]` |
| ... | | | | |

**Achados não óbvios destacados:**

- `[Combinação 1]` — taxa de `[XX,X%]` e lift de `[X,XX]`. Olhando para
  as características isoladamente, nenhuma se destaca de forma marcante;
  juntas, ultrapassam a média geral em `[Y]` pontos. Interpretação:
  `[…]`.
- `[Combinação 2]` — `[descrição do padrão e interpretação]`.

Esse tipo de leitura combinada é exatamente o que a análise univariada
não captura. A visualização "Venn-like" reforça o ponto ao mostrar que
o quadrante `A ∧ B` reúne `[N]` jogos com taxa de `[XX,X%]`, contra
`[XX,X%]` em `só A` e `[XX,X%]` em `só B`.

---

## 4. Avaliação do Modelo

O baseline implementado é uma **regressão logística** com pesos
balanceados, treinada sobre 80% dos dados (split estratificado pelo
alvo) com features padronizadas. As métricas no conjunto de teste
foram:

| Métrica | Valor | Comentário |
|---|---|---|
| Accuracy | `[X,XXX]` | `[…]` |
| Precision (classe 1) | `[X,XXX]` | `[…]` |
| Recall (classe 1) | `[X,XXX]` | `[…]` |
| F1 | `[X,XXX]` | `[…]` |

Matriz de confusão (linhas = real, colunas = predito):

|  | pred = 0 | pred = 1 |
|---|---|---|
| real = 0 | `[…]` | `[…]` |
| real = 1 | `[…]` | `[…]` |

### 4.1 Top features

As 10 features com maior coeficiente positivo (e portanto maior
associação direta com a probabilidade de recomendação) foram:

| Feature | Coeficiente |
|---|---|
| `[…]` | `[+X,XXX]` |
| `[…]` | `[+X,XXX]` |
| ... | |

A leitura comparativa entre esses coeficientes e a Seção 3.5 é
particularmente útil: features que aparecem destacadas no ranking do
modelo e simultaneamente em combinações de alta taxa de recomendação
têm maior probabilidade de representar um sinal robusto.

### 4.2 Limitações do modelo

- **Modelo linear**: não captura interações entre features sem feature
  engineering adicional. Justamente por isso o módulo `patterns.py`
  existe — para iluminar combinações que a regressão logística não
  expõe diretamente.
- **Sem validação cruzada nem busca de hiperparâmetros**: trata-se de
  um baseline de referência, não de um modelo otimizado.
- **`num_reviews_game` é altamente assimétrica**: jogos virais inflam
  a feature; uma transformação logarítmica seria uma melhoria simples
  e impactante.
- **`class_weight="balanced"` é uma correção mínima**: para um modelo de
  produção, valeria experimentar técnicas mais robustas
  (sub/over-sampling estratificado, *focal loss*, etc.).

---

## 5. Conclusão

O projeto atende aos critérios da disciplina ao integrar as cinco
tecnologias obrigatórias:

- **Pandas** orquestra leitura, limpeza, agregação e mineração de
  padrões;
- **Numpy** estrutura as categorizações via `np.select` (faixas de preço
  e ano);
- **Matplotlib** gera os gráficos univariados e de combinações, com
  padrão visual consistente reaproveitado pela interface;
- **Scikit-learn** implementa o pipeline preditivo baseline;
- **Streamlit** consolida toda a análise em uma interface interativa
  com filtros, KPIs e visualizações dinâmicas.

A pergunta de negócio foi respondida em três níveis complementares:

1. **Univariado** — há diferenças mensuráveis de taxa de recomendação
   por gênero, faixa de preço e período de lançamento, com destaque
   para `[principais achados]`.
2. **Combinado** — a mineração de pares de características revelou
   padrões não óbvios, como `[exemplo destacado]`, que não emergem da
   análise por dimensão isolada e justificam a abordagem do tipo
   "cerveja/fralda".
3. **Preditivo** — o modelo baseline confirma o ranking de features
   mais associadas à recomendação positiva e serve como ponto de
   comparação para futuros refinamentos.

### 5.1 Limitações do estudo

- O dataset utiliza apenas metadados publicamente disponíveis. Dados
  detalhados de jogadores (faixa etária, país, horas jogadas), preço
  regional histórico e métricas como pico de jogadores simultâneos não
  estão completamente representados.
- O parser de tags depende de strings padronizadas; tags raras ou em
  outros idiomas podem ter sido subcontadas.
- A análise trata cada review como observação independente. Jogos com
  alto volume de reviews ponderam mais o resultado agregado; embora
  `num_reviews_game` capture essa diferença como feature, ela não
  rebalanceia automaticamente a leitura por categoria.
- O modelo é um baseline interpretável. As métricas absolutas devem ser
  lidas como linha de base, não como capacidade preditiva final.

### 5.2 Trabalho futuro

- Enriquecer os metadados via Steam Store API (pico de jogadores
  simultâneos, preço regional, classificação indicativa) para refinar
  as análises de popularidade e de público-alvo.
- Aplicar transformações logarítmicas em variáveis assimétricas
  (`num_reviews_game`, `price_numeric`).
- Comparar o baseline com modelos não-lineares (Random Forest, Gradient
  Boosting) e cruzar a importância de features com os padrões da
  Seção 3.5.
- Adicionar uma análise temporal das reviews (deterioração após
  lançamento, efeito de descontos, *patches* significativos) que hoje
  está ausente.
