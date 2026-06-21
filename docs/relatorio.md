# Relatório — O que faz um jogo ser bem avaliado na Steam?

**Disciplina:** Análise de Dados e Solução de Problemas com Python
**Integrantes:** Alberth Cavalcanti, Ivo Pinheiro
**Repositório:** https://github.com/alber-th/new-technologies-project

> **Sobre os números.** Os valores numéricos abaixo são resultado da
> execução completa dos pipelines sobre o dataset enriquecido (192 jogos,
> 4.594.075 reviews) — `main_enrich.py` → `main_preprocessing.py` →
> `main_eda.py` → `main_patterns.py` → `main_modeling.py`. Para
> reproduzir, ver instruções em [`README.md`](../README.md).

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
- **Critério inicial de seleção:** alto volume de reviews com sinal
  binário de recomendação (`voted_up`) e identificação por jogo, o que
  permite agregações no nível de título.
- **Lacuna identificada após o download:** o dataset entrega **um CSV
  por jogo** (~192 arquivos, ~1,5 GB) contendo **apenas dados de
  review** — sem gênero, tags, preço, data de lançamento, desenvolvedor
  ou publisher. Exatamente as dimensões que a pergunta de negócio
  exige.
- **Resposta de engenharia:** uma etapa adicional de enriquecimento
  (Seção 2.2) busca esses metadados na Steam Store API pública e
  materializa um CSV alinhado por `app_id`. O pipeline analítico
  trabalha sobre a combinação reviews + metadados.
- **Organização em disco:** CSVs brutos em
  `data/raw/game_rvw_csvs/<app_id>_<Nome>.csv`; metadados enriquecidos
  em `data/raw/games_metadata.csv`; versão limpa e com features em
  `data/processed/steam_reviews_processed.csv`.

### 2.2 Enriquecimento via Steam Store API

`src/enrichment.py` implementa um cliente da API pública
`https://store.steampowered.com/api/appdetails`, sem autenticação. Para
cada `app_id` extraído dos nomes de arquivo, o módulo:

1. Faz uma requisição com `filters=basic,genres,categories,price_overview,release_date`
   para minimizar o payload.
2. Achata o JSON aninhado num registro plano com as colunas `app_name`,
   `genres`, `tags` (mapeadas a partir de `categories`), `release_date`,
   `price`, `developer`, `publisher`.
3. Concilia ausências de dados (jogos removidos da loja, fichas
   restritas) preservando o `app_id` com campos nulos — assim as
   reviews continuam contadas, apenas sem metadados.

O orquestrador `main_enrich.py` lê os arquivos em
`data/raw/game_rvw_csvs/`, extrai os `app_id` únicos e dispara a coleta
com **sleep de 1,5 s entre chamadas** (margem ampla frente ao rate
limit prático da Steam) e **idempotência via parâmetro `resume`** —
execuções subsequentes só buscam IDs ainda faltantes. A primeira
execução demora ~5 minutos; nas seguintes, é instantânea. O resultado
é persistido em `data/raw/games_metadata.csv`.

A documentação detalhada da etapa (decisões de engenharia, mapeamento
de schema, limitações e notas para apresentação) está em
[`docs/enriquecimento_steam_api.md`](./enriquecimento_steam_api.md).

### 2.3 Estrutura do projeto

```
src/
  enrichment.py       # cliente Steam Store API
  data_loading.py     # load_raw_data + load_combined_data
  preprocessing.py    # limpeza + feature engineering
  eda.py              # análise univariada
  patterns.py         # mineração de combinações ("cerveja/fralda")
  modeling.py         # baseline ML
app_streamlit.py      # interface interativa
main_*.py             # scripts que orquestram cada etapa
data/                 # raw e processed (ignorados pelo Git)
docs/                 # figuras, relatório e documentação técnica
```

Cada módulo é responsável por uma etapa do pipeline e devolve estruturas
reaproveitáveis (DataFrames, objetos `Figure`, modelos `Pipeline`),
permitindo que tanto os scripts `main_*.py` quanto a interface Streamlit
consumam a mesma lógica sem duplicação.

### 2.4 Carregamento e limpeza (Pandas)

`src/data_loading.py` expõe `load_combined_data`, que concatena os 192
CSVs de reviews (descartando o texto livre na leitura para conter
memória), renomeia colunas para o schema canônico (`voted_up` →
`recommended`, `recommendationid` → `review_id`, etc.), converte o
`timestamp_created` de unix epoch para `datetime` e faz `merge` com o
CSV de metadados em `app_id`. O resultado é um `DataFrame` único pronto
para o preprocessing.

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

### 2.5 Engenharia de features (Pandas + Numpy)

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

### 2.6 Análise exploratória (Matplotlib)

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

### 2.7 Mineração de padrões (Pandas)

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

### 2.8 Modelo baseline (Scikit-learn)

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

### 2.9 Interface (Streamlit)

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

### 3.1 Visão geral

- Total de reviews analisadas: **4.594.075**.
- Jogos únicos: **186** (de 192 — seis fichas estavam indisponíveis na
  Steam Store no momento da coleta).
- Taxa global de recomendação: **87,6%**.

A base é fortemente assimétrica em favor da classe positiva — quase
9 em cada 10 reviews recomendam o jogo. Esse fato metodológico
justifica o uso de **lift** (razão sobre o baseline) em vez da taxa
absoluta para ranquear padrões: variações de poucos pontos percentuais
já representam efeitos relevantes nesse regime.

### 3.2 Taxa de recomendação por gênero / modo

Calculando a taxa entre os jogos onde cada flag está ativa:

| Característica | Taxa | Reviews (n) |
|---|---|---|
| `has_indie` | **93,5%** | 1.789.257 |
| `has_singleplayer` | 90,0% | 3.578.971 |
| `has_rpg` | 88,8% | 1.413.397 |
| `has_coop` | 88,0% | 2.779.354 |
| `has_multiplayer` | 86,6% | 3.545.460 |
| `has_action` | **85,1%** | 3.002.923 |

Jogos **indie** são os mais bem avaliados (93,5%, quase 6 pontos acima
do baseline) — coerente com a noção de que projetos menores, sem o peso
de expectativas de uma franquia AAA, tendem a entregar mais
consistentemente o que prometem. Jogos majoritariamente **action** ou
focados em **multiplayer** ficam abaixo do baseline, refletindo
desafios típicos do gênero (curva de aprendizado, balanceamento,
servidores, comunidade tóxica).

### 3.3 Taxa de recomendação por faixa de preço

| Faixa | Taxa | Reviews (n) | Comentário |
|---|---|---|---|
| Grátis | **67,2%** | 303.508 | Forte penalidade — provável efeito de modelos *free-to-play* com monetização agressiva (lojas de skins, *gachas*, *pay-to-win*). |
| Barato (≤ R$ 20) | **94,0%** | 476.594 | Faixa de melhor recepção. Inclui muitos *indies* premium e bundles. |
| Médio (R$ 20–60) | 91,2% | 1.810.482 | Projetos consolidados de orçamento médio. |
| Caro (> R$ 60) | 86,0% | 1.838.678 | Próximo ao baseline. Lançamentos AAA são bem cobrados pela comunidade — expectativa alta penaliza qualquer fraqueza. |

A diferença Barato vs Grátis (mais de 26 pontos) é o achado mais
contundente da Seção 3 e contraria intuições ingênuas de que "preço
baixo significa qualidade baixa" — a relação é, na verdade, o oposto.

### 3.4 Taxa de recomendação por período de lançamento

| Período | Taxa | Reviews (n) |
|---|---|---|
| Antes de 2010 | **96,1%** | 187.660 |
| 2010–2015 | 91,1% | 1.334.495 |
| 2016–2020 | **85,3%** | 2.871.154 |
| Depois de 2020 | 88,0% | 192.089 |

Existe uma tendência **decrescente** clara entre os jogos mais antigos
e os mais recentes — atribuível a dois efeitos compostos: (i) **viés
de sobrevivência**: só permanecem na vitrine da Steam, com volume
relevante de reviews, os clássicos que entregaram valor; e (ii)
**inflação de catálogo** pós-2014, com explosão de lançamentos
(facilitada por Steam Direct) que diluiu a qualidade média. O leve
repique em "Depois de 2020" pode refletir a maturação do mercado indie
e o ecossistema mais robusto de plataformas pós-pandemia.

### 3.5 Padrões escondidos (combinações de 2 características)

A mineração com `min_games = 50` produziu o Top 10 abaixo, ranqueado
por taxa de recomendação. O **lift** mede o ganho da combinação sobre
o baseline da Steam (0,876).

| # | Combinação | Taxa | Lift | Nº jogos | Nº reviews |
|---|---|---|---|---|---|
| 1 | **singleplayer + indie** | **94,4%** | **1,08x** | 55 | 1.507.938 |
| 2 | multiplayer + singleplayer | 89,9% | 1,03x | 112 | 2.591.867 |
| 3 | singleplayer + coop | 89,5% | 1,02x | 80 | 2.135.491 |
| 4 | singleplayer + action | 88,9% | 1,01x | 81 | 2.170.796 |
| 5 | singleplayer + 2016-2020 | 88,8% | 1,01x | 102 | 1.939.308 |
| 6 | multiplayer + coop | 88,0% | 1,00x | 94 | 2.779.354 |
| 7 | singleplayer + Caro | 87,8% | 1,00x | 79 | 1.604.313 |
| 8 | action + coop | 86,3% | 0,99x | 60 | 2.116.929 |
| 9 | coop + 2016-2020 | 85,9% | 0,98x | 64 | 1.786.129 |
| 10 | multiplayer + action | 84,6% | 0,97x | 80 | 2.669.812 |

**Achado destacado — "singleplayer + indie".** Indie isolado já é a
melhor categoria por flag (93,5%); somado ao perfil singleplayer salta
para 94,4% com **lift 1,08x sobre o baseline**. A combinação cobre
1,5 milhão de reviews em 55 jogos únicos — base ampla e estatisticamente
robusta. **Interpretação:** o público indie singleplayer tende a ter
expectativas mais bem calibradas e maior tolerância a escopo limitado,
desde que a entrega seja coesa. É um perfil de baixo risco para
estúdios pequenos.

**Observação metodológica.** A distribuição de Steam é tão favorável à
classe positiva que poucas combinações ultrapassam lift 1,05x. Isso
não invalida o ranking — pelo contrário, mostra que a base é
"saturada de positividade" e a sinalização vem de quem se afasta
positivamente do baseline. As combinações com lift < 1,0x (action
combos, multiplayer + action) sinalizam o oposto: gêneros em que
desafios técnicos e expectativas mais altas penalizam consistentemente
a recepção.

---

## 4. Avaliação do Modelo

O baseline implementado é uma **regressão logística** com pesos
balanceados, treinada sobre 80% dos dados (3.675.260 reviews em treino,
918.815 em teste; split estratificado pelo alvo) com features
padronizadas. As 13 features finais foram: `has_*` (6 flags),
`price_numeric`, `release_year`, `num_reviews_game`,
`author.playtime_forever`, `author.num_reviews`, `received_for_free`,
`written_during_early_access`.

| Métrica | Valor | Comentário |
|---|---|---|
| Accuracy | **0,638** | Inferior ao naïve "prever 1 sempre" (87,6%), mas isso é esperado: o `class_weight="balanced"` força o modelo a errar mais positivos para captar negativos. |
| Precision (classe 1) | **0,930** | Quando o modelo diz "recomendado", acerta 93% das vezes. |
| Recall (classe 1) | **0,634** | Captura ~63% das reviews realmente positivas. |
| F1 | **0,754** | Compromisso saudável entre precision e recall. |

Matriz de confusão no conjunto de teste:

|  | pred = 0 | pred = 1 |
|---|---|---|
| real = 0 (não rec.) | 75.918 | 38.265 |
| real = 1 (rec.) | 294.683 | 509.949 |

Lendo a matriz: das 114.183 reviews realmente negativas, o modelo
classifica 75.918 (66%) corretamente — recall *na classe minoritária*
de aproximadamente 0,66, que é o que o balanceamento estava buscando
otimizar. Sem o balanceamento, o modelo cairia em "prever 1 sempre" e
o recall em "não recomendado" seria zero.

### 4.1 Top 10 features

Coeficientes da regressão logística sobre features padronizadas
(comparáveis entre si pela magnitude). Valor positivo aumenta a
probabilidade de recomendação:

| Feature | Coeficiente | Direção |
|---|---|---|
| `has_indie` | **+0,533** | ↑ |
| `has_singleplayer` | +0,174 | ↑ |
| `author.playtime_forever` | +0,111 | ↑ |
| `has_coop` | +0,067 | ↑ |
| `received_for_free` | +0,063 | ↑ |
| `has_multiplayer` | -0,050 | ↓ |
| `written_during_early_access` | -0,071 | ↓ |
| `has_rpg` | -0,084 | ↓ |
| `price_numeric` | -0,102 | ↓ |
| `has_action` | -0,141 | ↓ |

**Leitura conjunta com a Seção 3:**

- `has_indie` aparece como **a feature mais forte do modelo** e
  simultaneamente como o **gênero líder na análise univariada (93,5%)**
  e **integrante da combinação mais bem ranqueada na mineração
  ("singleplayer + indie", 94,4%)**. Esse cruzamento triplo é o sinal
  mais robusto do estudo.
- `price_numeric` com coeficiente negativo confirma o padrão
  contraintuitivo da Seção 3.3: **preço alto correlaciona-se
  negativamente com recomendação** quando isolado de outras features.
- `has_action` negativo (-0,141) reforça que o gênero, apesar de
  enorme em volume, tende a frustrar mais a expectativa do jogador
  médio do que outros nichos.
- `author.playtime_forever` positivo é esperado: quem joga muito tende
  a recomendar (selecionou jogos que gosta) — sinal mais sobre o autor
  do que sobre o jogo, mas útil de capturar como controle.

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

A pergunta de negócio foi respondida em três níveis complementares,
que convergem para o mesmo sinal:

1. **Univariado** — `has_indie` lidera as flags (93,5% vs 87,6% do
   baseline) e a faixa "Barato" lidera os preços (94,0% vs 67,2% do
   "Grátis"). Períodos anteriores a 2010 superam os mais recentes
   (96,1% vs 85,3% em 2016–2020), com efeito provavelmente combinado
   de sobrevivência editorial e inflação de catálogo pós-Steam Direct.
2. **Combinado** — a mineração revelou **"singleplayer + indie" como a
   combinação líder (94,4%, lift 1,08x sobre o baseline)**, cobrindo
   55 jogos únicos e 1,5 milhão de reviews — base estatisticamente
   robusta. Combinações com `action` e `multiplayer` ficam abaixo do
   baseline, sinalizando que esses gêneros enfrentam expectativas mais
   exigentes de balanceamento, servidores e comunidade.
3. **Preditivo** — o modelo baseline confirma o ranking: `has_indie`
   é a feature com maior coeficiente positivo (+0,533), seguida por
   `has_singleplayer` (+0,174). O cruzamento entre o ranking do modelo,
   o achado univariado e a combinação líder representa o sinal mais
   robusto identificado neste estudo.

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
