## Ideia 3 – O que faz um jogo ser “bem avaliado” na Steam?

**Pergunta de negócio**“Quais características (tags, ano de lançamento, gênero, etc.) estão associadas a reviews positivos/recomendados na Steam?”

**Datasets sugeridos**

- _Steam Game Reviews_ – reviews de ~200 jogos no Steam, incluindo texto de review e recomendação.[](https://www.kaggle.com/datasets/smeeeow/steam-game-reviews/code)
    
- _Sentiment Analysis for Steam Reviews_ / _Steam Game Review Dataset_ – dataset com reviews, campo de recomendação (0/1) e metadados do jogo (desenvolvedor, publisher, tags etc.).
    

**O que dá para fazer tecnicamente**

- **Definição do problema:** ajudar um estúdio a entender que tipos de jogos tendem a receber mais recomendações positivas.
    
- **Preparação dos dados:**
    
    - Tratar duplicatas de reviews, normalizar ano, agrupar tags em categorias (ex.: “RPG”, “Action”, “Indie”, “Multiplayer”).
        
    - Criar features binárias do tipo “tem multiplayer?”, “tem história forte?”, “tem modo cooperativo?” com base nas tags.
        
- **Análise exploratória + gráficos:**
    
    - Taxa de recomendação por gênero de jogo, por ano, por faixa de preço (se disponível).
        
    - Barplot com as 10 tags mais associadas a reviews recomendadas vs não recomendadas.
        
    - Wordcloud simples das palavras mais frequentes em reviews positivas vs negativas (pode ser só visual, sem NLP pesado).
        
- **Modelo (opcional):**
    
    - Classificador simples que usa só metadados (sem texto) para prever se um jogo será “recomendado” (baseline interessante).
        

**Padrões “tipo cerveja/fralda”**

- Combinações de tags não óbvias:
    
    - Ex.: jogos com tag “singleplayer + crafting + cozy” terem muito mais recomendação do que “singleplayer + crafting + survival”.
        
- Ver se jogos de determinado gênero só começam a ter reviews boas após um certo ano (ex.: melhoria de qualidade em indies de plataforma a partir de 2015).