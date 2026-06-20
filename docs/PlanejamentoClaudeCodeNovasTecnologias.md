
***

## Etapa 0 – Escolher dataset e criar estrutura do projeto

Objetivo: fixar qual dataset da Steam será usado, criar a estrutura de pastas/arquivos (src, data, app_streamlit.py, etc.) e documentar isso no repositório.[^2][^4][^6][^1]

**Prompt 0 – Setup inicial do projeto**

```text
Você é um assistente de desenvolvimento em Python focado em análise de dados.

Contexto do trabalho (resumo):
- Disciplina: Análise de Dados e Solução de Problemas com Python.
- Tecnologias obrigatórias: Numpy, Pandas, Matplotlib, Scikit-learn (modelo de ML opcional) e Streamlit.
- Entregáveis: código .py rodando de ponta a ponta, interface em Streamlit com gráficos, e relatório em PDF.

Tema do projeto:
- Ideia 3 – “O que faz um jogo ser bem avaliado na Steam?”
- Pergunta de negócio: “Quais características (tags, ano de lançamento, gênero, etc.) estão associadas a reviews positivos/recomendados na Steam?”.

Datasets candidatos (Kaggle):
- Steam Game Review Dataset: https://www.kaggle.com/datasets/arashnic/game-review-dataset
- Steam Game Reviews: https://www.kaggle.com/datasets/smeeeow/steam-game-reviews
- Sentiment Analysis for Steam Reviews: https://www.kaggle.com/datasets/piyushagni5/sentiment-analysis-for-steam-reviews

Quero que você:
1) Analise rapidamente a descrição desses 3 datasets (sem baixar ainda) e sugira QUAL deles é mais adequado para:
   - Ter campo de recomendação (positivo/negativo) ou similar.
   - Ter metadados de jogo (tags, gênero, ano, desenvolvedor/publisher, etc.).
   - Permitir responder à pergunta de negócio com foco em metadados, sem precisar de NLP pesado no texto das reviews.
2) Sugira a estrutura de pastas e arquivos para o projeto, por exemplo:
   - data/raw
   - data/processed
   - src/
       - data_loading.py
       - preprocessing.py
       - eda.py
       - modeling.py (opcional)
       - viz_utils.py
   - app_streamlit.py
   - notebooks/ (se achar útil)
   - docs/
3) Gere o código mínimo necessário (em Python) para:
   - Criar essa estrutura de pastas (se for rodar localmente).
   - Criar um arquivo README.md com um parágrafo explicando o objetivo do projeto.

IMPORTANTE:
- Não escreva ainda código de análise de dados; foque só em escolha do dataset + estruturação inicial de projeto.
- Me entregue:
  - Uma decisão clara de qual dataset vamos usar.
  - Um bloco de código Python que eu possa rodar para criar a estrutura de diretórios.
  - O conteúdo sugerido inicial do README.md.
```


***

## Etapa 1 – Definição formal do problema e documentação

Objetivo: transformar a ideia em uma seção formal de “Definição do Problema” para usar no relatório e como comentário no código principal.[^3][^1][^2]

**Prompt 1 – Definir problema de negócio e objetivos analíticos**

```text
Agora considere que já escolhemos o dataset da Steam (aquele definido na etapa anterior).

Contexto adicional:
- Do arquivo "Ideia-base-do-Projeto.md": queremos entender que tipos de jogos tendem a receber mais recomendações positivas na Steam, usando tags, gêneros, ano, etc., e identificar combinações não óbvias de características (padrões tipo “cerveja/fralda”). 
- Do arquivo "Pontos-a-se-Analisar.md": queremos olhar para gêneros, avaliações e quantidade de avaliações, dados de usuários (quando houver), público-alvo (faixa etária), data de lançamento, estúdio/empresa, país, pico de jogadores simultâneos, preço (considerando preço regional), e também relações pouco óbvias entre esses fatores.

Tarefas:
1) Escrever uma **Definição do Problema** clara e objetiva em 2–3 parágrafos, em português, para colocar no relatório:
   - Explicar o contexto (Steam, avaliações de jogos).
   - Explicar o problema de negócio (ajudar um estúdio/analista a entender o que faz um jogo ser bem avaliado).
   - Explicar o valor de encontrar padrões não óbvios (tipo correlações entre combinações de tags, preço, ano de lançamento, etc.).
2) Listar de forma estruturada (bullets) os **Objetivos Analíticos**:
   - Exemplos: medir taxa de recomendação por gênero, ano, faixa de preço; identificar combinações de tags associadas a maior taxa de recomendação; etc.
3) Produzir um texto curto (em Markdown) para ser inserido no README.md e no relatório (seção “Introdução ao Problema”).

Entregue:
- Texto da Definição do Problema.
- Lista dos Objetivos Analíticos.
- Versão curta para README.md.
Não escreva código nesta etapa, apenas texto.
```


***

## Etapa 2 – Download, carregamento e inspeção inicial do dataset

Objetivo: baixar o dataset escolhido, salvar em `data/raw`, carregar com Pandas, inspecionar colunas, tipos, tamanhos e nulos.[^4][^5][^6]

**Prompt 2 – Carregar dados e fazer diagnóstico inicial**

```text
Agora vamos começar a codar em Python.

Contexto:
- Já escolhemos o dataset da Steam na Etapa 0.
- O arquivo .csv (ou similar) será colocado em data/raw/ com um nome fixo, por exemplo: data/raw/steam_reviews.csv.

Tarefas:
1) Criar um módulo src/data_loading.py com funções:
   - load_raw_data(path: str) -> pd.DataFrame
   - show_basic_info(df: pd.DataFrame) -> None
2) O código deve:
   - Usar Pandas para carregar o dataset.
   - Exibir: número de linhas/colunas, tipos de dados, contagem de nulos por coluna, amostra de 5–10 linhas.
3) No final, no arquivo principal (por exemplo, um script `main_diagnostico.py`), chamar essas funções para:
   - Carregar o dataset cru de data/raw.
   - Imprimir as informações básicas no console.

Requisitos:
- Utilizar apenas bibliotecas padrão + pandas + numpy.
- Escrever código limpo, com funções bem nomeadas, comentários mínimos porém claros.
- Separar bem o módulo de carregamento (data_loading.py) do script de execução.

Entregue:
- Código completo de data_loading.py.
- Código completo de main_diagnostico.py com exemplo de uso.
```


***

## Etapa 3 – Limpeza, tratamento e criação de features

Objetivo: tratar nulos/duplicatas, converter tipos, criar features derivadas (ex.: dummies de tags, faixas de preço, faixas de ano, etc.), alinhado ao Critério 1.[^1][^2][^3]

**Prompt 3 – Preprocessing e engenharia de atributos**

```text
Vamos implementar a etapa de obtenção e preparação dos dados.

Contexto:
- Temos um DataFrame com os dados crus de reviews da Steam, incluindo:
  - Alguma forma de "recomendação" (ex.: recommended = True/False, ou score positivo/negativo).
  - Metadados do jogo: nome, data de lançamento, desenvolvedor, publisher, tags/gênero, preço, etc. (adapte ao dataset real).
- Queremos preparar uma base limpa e enriquecida em data/processed/steam_reviews_processed.csv.

Tarefas:
1) Criar um módulo src/preprocessing.py com funções, por exemplo:
   - clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
       - Remover duplicatas óbvias (por id de review ou combinação jogo+usuário+data).
       - Tratar nulos (decidir quando descartar linha ou preencher com valor padrão).
       - Converter colunas de data para datetime.
       - Converter coluna de recomendação para binária (1 = recomendado, 0 = não recomendado).
   - engineer_features(df: pd.DataFrame) -> pd.DataFrame:
       - Criar faixas de preço (barato, médio, caro) com base em quantis ou regras simples.
       - Criar faixas de ano de lançamento (antes de 2010, 2010–2015, 2016–2020, depois de 2020).
       - Transformar tags/gêneros em variáveis binárias simples, por exemplo: has_multiplayer, has_singleplayer, has_rpg, has_action, has_coop, has_indie (use parsing de strings de tags).
       - Calcular, se possível, uma medida de popularidade do jogo (ex.: número de reviews por jogo).
2) Criar um script main_preprocessing.py que:
   - Carrega o dataset cru usando data_loading.load_raw_data.
   - Aplica clean_raw_data e engineer_features.
   - Salva o resultado em data/processed/steam_reviews_processed.csv.
   - Mostra um resumo das novas colunas criadas.

Requisitos:
- Usar Numpy quando fizer sentido (por exemplo, para criar faixas com np.select ou np.where).
- Garantir que as transformações sejam bem comentadas para justificar o tratamento de dados.
- Código modular, com funções reutilizáveis.

Entregue:
- Código completo de src/preprocessing.py.
- Código completo de main_preprocessing.py.
```


***

## Etapa 4 – Análise exploratória e visualizações básicas (sem Streamlit ainda)

Objetivo: criar funções de EDA com Matplotlib para explorar taxa de recomendação por gênero, ano, preço, etc., preparando os gráficos que depois irão para o Streamlit.[^2][^3][^1]

**Prompt 4 – EDA com Matplotlib (modo notebook/script)**

```text
Agora vamos fazer a análise exploratória dos dados processados.

Contexto:
- Temos data/processed/steam_reviews_processed.csv com:
  - Coluna binária de recomendação (ex.: recommended_bin).
  - Colunas de faixas de preço, faixas de ano, dummies de gêneros/tags, popularidade (nº de reviews), etc.

Tarefas:
1) Criar um módulo src/eda.py com funções, por exemplo:
   - plot_recommendation_rate_by_genre(df)
   - plot_recommendation_rate_by_price_range(df)
   - plot_recommendation_rate_by_release_period(df)
   - plot_top_genres_by_review_count(df)
   Cada função deve:
   - Agrupar os dados de forma adequada.
   - Calcular taxa de recomendação (média de recommended_bin) por categoria.
   - Gerar gráfico de barras com Matplotlib, com título, rótulo de eixos e legendas claras.
2) Criar um script main_eda.py que:
   - Carrega o dataset processado.
   - Chama essas funções gerando e salvando os gráficos em uma pasta, por exemplo: docs/figures/.
   - Mostra alguns prints com estatísticas resumo (ex.: top 5 gêneros com maior taxa de recomendação).

Requisitos:
- Utilizar Matplotlib diretamente (sem Seaborn, a não ser que a disciplina permita).
- Garantir que os gráficos terão boa leitura quando forem inseridos na interface Streamlit.
- Não usar ainda Streamlit; esta etapa é para validar a lógica analítica.

Entregue:
- Código completo de src/eda.py.
- Código completo de main_eda.py.
```


***

## Etapa 5 – Buscar “padrões escondidos” e combinações tipo “cerveja/fralda”

Objetivo: ir além do básico, procurando relações não óbvias entre combinações de tags, faixa de preço, ano, etc., como vocês mencionaram nos “Pontos a se analisar”.[^3][^1]

**Prompt 5 – Análise de combinações de características**

```text
Agora quero focar em descobrir padrões menos óbvios, tipo “cerveja/fralda”, usando a base processada da Steam.

Contexto:
- Temos colunas de:
  - Gênero/tags binárias (has_rpg, has_action, has_indie, has_coop, has_multiplayer, etc.).
  - Faixas de preço.
  - Faixas de ano de lançamento.
  - Popularidade (número de reviews do jogo).
  - Taxa de recomendação (recommended_bin).

Tarefas:
1) No módulo src/eda.py (ou em um novo módulo src/patterns.py), implementar funções para:
   - Calcular taxa de recomendação para **combinações de 2 características**, por exemplo:
     - (gênero principal, faixa de preço)
     - (gênero principal, faixa de ano)
     - (tem_coop, tem_multiplayer)
   - Encontrar as top N combinações com maior taxa de recomendação, filtrando apenas combinações com número mínimo de jogos (ex.: pelo menos 50 jogos).
2) Implementar uma função que gere uma tabela ou gráfico de barras mostrando:
   - Top 10 combinações (ex.: "RPG + barato", "Indie + cozy + singleplayer") com maior taxa de recomendação.
3) (Opcional) Criar alguma visualização que lembre um “diagrama de Venn simplificado” usando proporções de jogos que pertencem a dois conjuntos (por exemplo, jogos que são both indie e coop vs indie only vs coop only), mesmo que não seja um Venn perfeito.

Requisitos:
- Focar em lógica de agrupamento com Pandas (groupby, size, mean).
- Garantir que haja filtros para não mostrar combinações com pouca amostra.
- Preparar as saídas (dataframes e gráficos) pensando em reaproveitar na interface Streamlit depois.

Entregue:
- Código das funções de busca de padrões (em eda.py ou patterns.py).
- Um script main_patterns.py que rode essas análises e imprima/mostre as top combinações encontradas.
```


***

## Etapa 6 – Modelo preditivo simples (opcional, mas recomendado)

Objetivo: criar um modelo básico (ex.: regressão logística ou árvore) que, dado metadados do jogo, prediz se ele tende a ser recomendado ou não.[^1][^2]

**Prompt 6 – Modelo de classificação para recomendação**

```text
Vamos criar um modelo de Machine Learning simples e bem documentado para prever se um jogo será bem avaliado (recomendado) com base em metadados.

Contexto:
- DataFrame processado com:
  - recommended_bin como variável alvo.
  - Features numéricas e binárias (preço, faixas de ano, popularidade, dummies de gênero/tags, etc.).

Tarefas:
1) Criar um módulo src/modeling.py com funções:
   - build_train_test_split(df) -> X_train, X_test, y_train, y_test
   - train_baseline_model(X_train, y_train) -> modelo
   - evaluate_model(model, X_test, y_test) -> dict com métricas (accuracy, precision, recall, matriz de confusão).
2) Utilizar um modelo simples do scikit-learn, por exemplo:
   - LogisticRegression ou RandomForestClassifier.
3) Criar um script main_modeling.py que:
   - Carrega o dataset processado.
   - Separa treino e teste.
   - Treina o modelo.
   - Imprime as métricas.
   - (Opcional) Calcula importância de features, se o modelo permitir, e mostra as top 10 features associadas a maior chance de recomendação.

Requisitos:
- Usar Scikit-learn.
- Código bem modular e legível, adequando-se ao nível de Engenharia de Software.
- Deixar claro que o modelo é um “baseline” simples, com limitações.

Entregue:
- Código completo de src/modeling.py.
- Código de main_modeling.py.
```


***

## Etapa 7 – Interface em Streamlit com visualizações (Critério 2)

Objetivo: criar a aplicação Streamlit que consome os dados processados e as funções de EDA, exibindo gráficos e tabelas interativas.[^2]

**Prompt 7 – Construir a interface Streamlit**

```text
Agora vamos construir a interface Streamlit, peça central do Critério 2.

Contexto:
- Temos módulos prontos:
  - data_loading.py
  - preprocessing.py
  - eda.py (e patterns.py se criado)
  - modeling.py (opcional)
- Temos data/processed/steam_reviews_processed.csv como base principal para visualização.

Tarefas:
1) Criar o arquivo app_streamlit.py com a seguinte estrutura geral:
   - Sidebar:
     - Filtros por gênero/tags principais (checkboxes ou multiselect).
     - Filtro por faixa de preço.
     - Filtro por faixa de ano.
   - Seções na página:
     1. Introdução: breve texto explicando o objetivo do app e do projeto.
     2. Visão geral:
        - KPIs simples: nº de jogos, taxa média de recomendação, gênero mais frequente, etc.
     3. Gráficos principais (reutilizando funções de eda.py):
        - Taxa de recomendação por gênero.
        - Taxa de recomendação por faixa de preço.
        - Taxa de recomendação por período de lançamento.
     4. Padrões escondidos:
        - Tabela/gráfico com top combinações de características (resultado de patterns.py).
     5. (Opcional) Modelo:
        - Exibir métricas do modelo (accuracy, etc.).
        - Talvez um pequeno formulário para o usuário escolher características e ver uma predição (se for simples de implementar).

2) Utilizar Matplotlib dentro do Streamlit:
   - Gerar as figuras com Matplotlib e exibir com st.pyplot(fig).

Requisitos:
- A aplicação deve rodar com `streamlit run app_streamlit.py` sem erros.
- Os gráficos precisam ter títulos, legendas e eixos claros.
- Lembrar de não recarregar o dataset a cada interação desnecessariamente (usar st.cache_data/st.cache_resource se estiver permitido pela versão).

Entregue:
- Código completo de app_streamlit.py.
```


***

## Etapa 8 – Geração do relatório em PDF (texto base)

Objetivo: produzir o texto base do relatório (Introdução, Metodologia, Resultados Visuais, Avaliação do Modelo, Conclusão) para depois ser formatado em PDF.[^3][^1][^2]

**Prompt 8 – Redigir relatório breve do projeto**

```text
Agora preciso que você me ajude a redigir o relatório breve em texto (para depois eu formatar em PDF).

Contexto:
- O projeto já foi implementado:
  - Definição do problema (Steam, jogos bem avaliados).
  - Pipeline de dados (carregamento, limpeza, criação de features).
  - EDA e interface Streamlit com gráficos de recomendação por gênero, preço, ano, etc.
  - Análise de combinações de características (padrões escondidos).
  - (Opcional) Modelo preditivo simples.

Tarefas:
1) Produzir um texto em português, em Markdown, com as seguintes seções:
   - Introdução ao Problema
   - Metodologia
   - Resultados Visuais
   - Avaliação do Modelo (se tiver modelo; se não, explicar que foi opcional e não implementado)
   - Conclusão
2) Em cada seção:
   - Descrever de forma objetiva o que foi feito, conectando com os requisitos da disciplina (uso de Pandas, Numpy, Matplotlib, Scikit-learn, Streamlit).
   - Explicar de maneira clara os principais achados, especialmente:
     - Fatores mais associados a jogos bem avaliados (por gênero, preço, ano, tags).
     - Padrões não óbvios encontrados nas combinações.
   - Citar limitações do estudo (dataset, ausência de dados de jogador, simplificações).

3) O texto deve ser direto e adequado como relatório de um trabalho de disciplina de Engenharia de Software, sem linguagem excessivamente informal.

Entregue:
- O texto completo em Markdown (sem necessidade de gerar o PDF; isso eu farei depois).
```


***

## Etapa 9 – Refino de código e organização para entrega

Objetivo: revisar o código, remover redundâncias, organizar imports, comentários e garantir que tudo rode “de ponta a ponta” sem erro, conforme Critério 3.[^2]

**Prompt 9 – Revisão final de código e checklist de entrega**

```text
Para fechar o projeto, quero uma revisão geral.

Contexto:
- Já temos:
  - Módulos em src/ (data_loading, preprocessing, eda, patterns, modeling).
  - Scripts main_*.py que executam cada etapa.
  - app_streamlit.py funcional.
  - Relatório em texto (Markdown) pronto para ser transformado em PDF.

Tarefas:
1) Fazer uma revisão conceitual do código (pode ser descritiva, sem analisar arquivo por arquivo):
   - Sugerir melhorias de organização (pastas, nomes de arquivos/módulos).
   - Sugerir padrões para nomes de funções e variáveis.
   - Apontar pontos em que há repetição de lógica que podem ser extraídos para funções auxiliares.
2) Propor um “roteiro de execução” para o avaliador, por exemplo:
   - Passo 1: rodar main_preprocessing.py
   - Passo 2: rodar main_eda.py
   - Passo 3: rodar main_patterns.py
   - Passo 4: rodar main_modeling.py (se modelo existir)
   - Passo 5: `streamlit run app_streamlit.py`
3) Sugerir um checklist final de entrega contendo:
   - Arquivos necessários.
   - Pastas que devem estar presentes.
   - Como anexar os dados e o PDF no AVA.
4) (Opcional) Sugerir pequenas melhorias de docstring/comentários para deixar o código mais legível.

Entregue:
- Uma lista estruturada (em texto) com:
  - Sugestões de refino de código.
  - Roteiro de execução.
  - Checklist de entrega.
```


***

Com esses prompts, vocês conseguem ir construindo o projeto em blocos bem separados, facilitando o versionamento (por exemplo, um commit por etapa/prompt) e garantindo que todos os critérios da disciplina sejam atendidos (problema bem formulado, tratamento de dados robusto, interface Streamlit clara, código limpo e relatório coerente).[^6][^4][^1][^3][^2]

<div align="center">⁂</div>

[^1]: Ideia-base-do-Projeto.md

[^2]: Instrucoes-do-Projeto.md

[^3]: Pontos-a-se-Analisar.md

[^4]: https://www.kaggle.com/datasets/smeeeow/steam-game-reviews/code

[^5]: https://www.kaggle.com/datasets/piyushagni5/sentiment-analysis-for-steam-reviews/tasks

[^6]: https://www.kaggle.com/datasets/arashnic/game-review-dataset

[^7]: https://www.kaggle.com/datasets/arashnic/game-review-dataset/versions/1

