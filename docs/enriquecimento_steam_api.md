# Enriquecimento de Metadados via Steam Store API

Este documento explica em detalhe por que o projeto precisou adicionar
uma etapa de enriquecimento, como ela foi implementada e quais decisões
de engenharia foram tomadas no caminho. É pensado para servir de apoio
durante a apresentação do trabalho.

---

## 1. Por que essa etapa existe

A escolha original de dataset foi **`smeeeow/steam-game-reviews`**
(Kaggle), por combinar o sinal de recomendação (`voted_up`) com volume
suficiente para análise quantitativa. O download confirmou as
expectativas no quesito **volume** (~192 jogos, ~milhões de reviews,
~1,5 GB), mas trouxe uma surpresa estrutural:

- O dataset entrega **um CSV por jogo**, nomeado no padrão
  `<app_id>_<NomeDoJogo>.csv` (ex.: `10_CounterStrike.csv`,
  `105600_Terraria.csv`).
- Cada CSV contém **apenas dados de review** (id, voto, timestamp,
  idioma, tempo de jogo do autor, etc.).
- **Nenhuma coluna de metadado do jogo**: não há gênero, tags,
  preço, data de lançamento ou estúdio.

Essa lacuna é incompatível com a tese central do projeto, que é
investigar **como características do jogo** (gênero, faixa de preço,
ano, etc.) se associam à taxa de recomendação — e em particular, buscar
combinações não óbvias do tipo "indie + barato → recomendação alta".

A saída escolhida foi **enriquecer o dataset original** com chamadas à
**Steam Store API pública**, recompondo no nosso schema os campos que
o Kaggle deixou de fora. Essa decisão preserva tudo o que já tinha sido
construído (preprocessing, EDA, patterns, modelo, Streamlit, relatório)
e mantém o foco analítico original.

---

## 2. A Steam Store API

A Steam expõe um endpoint público não-autenticado para consultar a
ficha de uma loja:

```
https://store.steampowered.com/api/appdetails?appids=<id>&filters=<lista>
```

**Características relevantes:**

| Aspecto | Como tratamos |
|---|---|
| Autenticação | Não exige. Não usamos chave de API. |
| Formato | JSON aninhado: `{ "<id>": { "success": bool, "data": {...} } }` |
| `success == false` | Jogo removido da loja / fichas restritas. Tratado como "sem metadados" — a linha entra com `NaN` mas o `app_id` é mantido para preservar as reviews. |
| Rate limit | Não documentado. Convenção da comunidade aponta ~200 req / 5 min. Para ficar bem abaixo disso, usamos **sleep configurável (default 1,5s) entre chamadas**. |
| Filtros | Aceita `filters=basic,genres,categories,...` para reduzir payload. Usamos `basic,genres,categories,price_overview,release_date` — corta descrições longas e screenshots. |
| Erros transitórios | Capturamos `requests.RequestException` por chamada, registramos no log de progresso e seguimos com `NaN` para aquele `app_id`. |

**Por que sequencial e não paralelo?** Ser educado com a API pública é
o caminho mais simples de evitar `429` / banimento temporário; com 192
jogos, ~5 min de coleta total é completamente aceitável e roda uma vez
só (idempotente).

---

## 3. Mapeamento de schema

A Steam retorna campos com nomes próprios e estrutura aninhada. Esta é
a tabela de "de → para" entre a resposta da API e o schema canônico
esperado por `src/preprocessing.py`:

| Campo JSON da API | Coluna no nosso schema | Tratamento |
|---|---|---|
| `data.name` | `app_name` | string direta |
| `data.genres[*].description` | `genres` | concatenação `", "` |
| `data.categories[*].description` | `tags` | concatenação `", "` (mapeia "Single-player", "Multi-player", "Co-op", etc.) |
| `data.release_date.date` | `release_date` | string `"MMM DD, YYYY"`, parseada depois pelo preprocessing |
| `data.is_free` (true) | `price = 0.0` | "free-to-play" vira preço zero |
| `data.price_overview.final` | `price` | em cents → divide por 100 |
| (sem `price_overview` e não free) | `price = NaN` | jogo sem preço público (DLC restrito, removido) |
| `data.developers` | `developer` | concatenação `", "` |
| `data.publishers` | `publisher` | concatenação `", "` |

**Por que `categories` vira `tags` no nosso schema?** As `categories`
da Steam são tags estruturadas de gameplay ("Single-player",
"Multi-player", "Co-op", "Online PvP") — exatamente o sinal que
alimenta nossas flags `has_multiplayer`, `has_singleplayer`,
`has_coop`. Os `genres` continuam separados (Action, RPG, Indie, etc.)
para alimentar `has_action`, `has_rpg`, `has_indie`. O parser de
`preprocessing._parse_tags` consegue extrair os tokens de ambas as
strings; a flexibilidade está no parser, não no formato de entrada.

---

## 4. Implementação

A etapa é dividida em dois artefatos:

### 4.1 `src/enrichment.py` — biblioteca

Quatro funções públicas, cada uma com responsabilidade única:

```python
extract_app_id_from_filename(path)  # "10_CounterStrike.csv" → 10
fetch_app_details(app_id, session)   # 1 chamada HTTP → dict | None
parse_app_details(data, app_id)      # JSON aninhado → linha plana
enrich_games(app_ids, output_csv)    # orquestrador, escreve CSV
```

**Decisões de design:**

- **`fetch_app_details` é "pura no erro"**: ele só decide entre
  "tem dados" ou "não tem", e propaga exceções de rede para o caller.
  Quem decide retry/log/skip é o `enrich_games`. Essa separação torna
  `fetch_app_details` trivialmente testável.
- **`parse_app_details(None, app_id)` devolve uma linha "vazia mas
  válida"** — com o `app_id` correto e os demais campos como `None`.
  Isso garante que jogos sem ficha pública continuam representados na
  base e suas reviews seguem contadas (apenas sem metadados).
- **Idempotência via `resume=True`**: se `games_metadata.csv` já
  existe, `enrich_games` lê os `app_id` já presentes e refaz a chamada
  apenas para os faltantes. Permite parar com `Ctrl+C` e retomar sem
  perder progresso.
- **`requests.Session()`** reutiliza a conexão HTTP entre chamadas —
  reduz latência em ~30% comparado a `requests.get` solto.
- **User-Agent customizado** ajuda em troubleshooting do lado da
  Steam (se eles bloquearem, sabem identificar a fonte).

### 4.2 `main_enrich.py` — script de execução

```bash
python main_enrich.py
```

O script faz:

1. Verifica se `data/raw/game_rvw_csvs/` existe (mensagem amigável se
   não).
2. Itera os arquivos `*.csv` e extrai `app_id`s únicos.
3. Avisa se um `games_metadata.csv` parcial já existe (vai retomar).
4. Chama `enrich_games(...)` passando um `on_progress` callback que
   imprime cada chamada como `[i/total] app_id=X status`.
5. No final, imprime quantos jogos vieram com metadados completos vs.
   quantos a API não tinha.

Saída: **`data/raw/games_metadata.csv`** com colunas
`[app_id, app_name, genres, tags, release_date, price, developer, publisher]`.

---

## 5. Mudanças no `src/data_loading.py`

A função antiga `load_raw_data(path)` continua existindo (útil para
inspecionar um CSV isolado em diagnósticos pontuais), mas o pipeline
analítico passa a usar a nova função:

```python
load_combined_data(reviews_dir, metadata_path) -> pd.DataFrame
```

O que ela faz:

1. **Concatena** todos os `*.csv` em `reviews_dir`, lendo apenas as
   colunas necessárias (`usecols=lambda c: c in _REVIEW_USECOLS_DEFAULT`)
   para conter o uso de memória — o texto livre da review é descartado
   na leitura, economizando algo em torno de 80% do tamanho original.
2. Para cada arquivo, extrai o `app_id` do **nome de arquivo** e
   adiciona como coluna.
3. **Renomeia** colunas do schema bruto para o canônico:

   | Bruto (Steam Kaggle) | Canônico (preprocessing) |
   |---|---|
   | `recommendationid` | `review_id` |
   | `voted_up` | `recommended` |
   | `timestamp_created` | `review_date` |
   | `author.steamid` | `author_id` |

4. **Converte** `review_date` de unix epoch para `datetime` via
   `pd.to_datetime(..., unit="s", errors="coerce")`. O preprocessing
   já espera datetime nessa coluna.
5. **Faz `merge(metadata, on='app_id', how='left')`** com o CSV de
   metadados. Jogos sem ficha pública entram com `NaN` nessas colunas
   — o `clean_raw_data` lida com isso preenchendo `"Unknown"`.

O resultado é um `DataFrame` único, no schema esperado, pronto para
entrar em `clean_raw_data` → `engineer_features`.

---

## 6. Fluxo completo (visão de execução)

```text
+--------------------------------------+
|  data/raw/game_rvw_csvs/             |
|    10_CounterStrike.csv              |
|    105600_Terraria.csv               |
|    ... (192 arquivos)                |
+--------------------------------------+
                 │
                 │  main_enrich.py
                 │  (1× ~5 min)
                 ▼
+--------------------------------------+
|  Steam Store API                     |
|  appdetails?appids=<id>              |
+--------------------------------------+
                 │
                 ▼
+--------------------------------------+
|  data/raw/games_metadata.csv         |
|  app_id, app_name, genres, tags,     |
|  release_date, price, developer,     |
|  publisher                           |
+--------------------------------------+
                 │
                 │  main_preprocessing.py
                 │   └─ load_combined_data()
                 │       ├─ concat dos 192 CSVs
                 │       ├─ rename → schema canônico
                 │       └─ merge on app_id
                 ▼
+--------------------------------------+
|  DataFrame combinado                 |
|  (reviews + metadata)                |
+--------------------------------------+
                 │
                 │  clean_raw_data() + engineer_features()
                 ▼
+--------------------------------------+
|  data/processed/                     |
|    steam_reviews_processed.csv       |
+--------------------------------------+
                 │
        ┌────────┴─────────────┐
        ▼                      ▼
   main_eda.py            app_streamlit.py
   main_patterns.py
   main_modeling.py
```

---

## 7. Limitações conhecidas

- **Preço em USD único.** Usamos `cc=us` implícito (default da API), o
  que pode esconder estratégias regionais de precificação (BRL com
  desconto regional típico da Steam). Trabalho futuro: amostrar
  `cc=br` ou múltiplos países e comparar.
- **Snapshot único de preço e dados.** O preço retornado é o atual,
  não o preço no momento da review. Jogos em promoção no momento da
  coleta podem aparecer artificialmente baratos.
- **Jogos removidos.** Alguns `app_id` deixam de existir na loja (jogos
  banidos, DLCs antigos, jogos que viraram free-to-play e mudaram
  status). Esses entram com metadados vazios e ficam visíveis no
  resumo final do `main_enrich.py`.
- **Categorias ≠ tags do usuário.** A Steam tem tanto `categories`
  (oficiais, fixas) quanto `user-defined tags` (mais granulares,
  tipo "Roguelike", "Cozy", "Souls-like"). O endpoint usado aqui
  retorna apenas `categories` — `tags` reais do usuário só são
  acessíveis via APIs internas não documentadas. Para o nível de
  análise do trabalho, `categories` + `genres` cobrem bem.
- **Sem versionamento da resposta.** Reexecutar o script no futuro
  pode trazer dados ligeiramente diferentes (preço atualizado,
  publisher mudou). Para reprodutibilidade plena, o
  `games_metadata.csv` deve ser preservado uma vez gerado.

---

## 8. Notas para a apresentação

Pontos que vale destacar ao mostrar o projeto:

1. **Por que houve enriquecimento**: o dataset Kaggle entregou reviews,
   não jogos. Nossa pergunta de negócio era sobre características do
   jogo. Identificamos o gap e decidimos cobri-lo via API pública sem
   trocar o dataset (que era bom em volume).
2. **Por que API pública e não outro Kaggle**: garante atualidade dos
   metadados, autonomia (sem depender de outro fornecedor de dados) e
   alinhamento exato com o conjunto de `app_id` que já temos.
3. **Idempotência como decisão de engenharia**: 5 min de coleta + risco
   de queda de rede justificam o `resume=True`. É padrão em qualquer
   pipeline que toca em API externa.
4. **Separação de responsabilidades**: `enrichment.py` é a biblioteca
   pura; `main_enrich.py` é a orquestração; `data_loading.py` é a
   ponte com o pipeline existente. Cada arquivo tem uma razão para
   mudar isolada das outras.
5. **Schema canônico estável**: o resto do pipeline (preprocessing,
   EDA, patterns, modelo, Streamlit) não precisou de nenhuma mudança
   porque o enriquecimento entrega exatamente as colunas que ele já
   esperava. É um exemplo prático de baixo acoplamento.
