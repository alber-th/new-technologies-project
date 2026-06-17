# new-technologies-project

Trabalho de Novas Tecnologias do 1º Semestre de 2026.
Integrantes: Alberth Cavalcanti e Ivo Pinheiro.

## O que faz um jogo ser bem avaliado na Steam?

Projeto da disciplina **Análise de Dados e Solução de Problemas com Python**.
O objetivo é investigar quais características de um jogo na Steam — gênero, tags,
ano de lançamento, desenvolvedor/publisher, entre outras — estão associadas a
reviews positivas/recomendadas pelos jogadores. A análise utiliza Numpy, Pandas,
Matplotlib e Scikit-learn, com uma interface interativa em Streamlit para
exploração dos resultados e um relatório em PDF como entregável final.

## Introdução ao Problema

A Steam concentra um volume massivo de avaliações de jogadores, em que cada
review é classificada como **Recomendada** ou **Não Recomendada**. Esse sinal
é, hoje, uma das principais referências de qualidade percebida de um jogo —
mas não é trivial entender, de forma quantitativa, **o que faz um jogo ser
bem avaliado**.

Este projeto investiga como características de metadado dos jogos — gênero,
tags, ano de lançamento, faixa de preço, estúdio/publisher, país de origem,
classificação indicativa e pico de jogadores simultâneos — se relacionam com
a **taxa de recomendação**. Mais do que olhar cada dimensão isoladamente,
o objetivo é encontrar **combinações não óbvias** de características
(no espírito do clássico "cerveja e fralda" do varejo) que se associam a
jogos consistentemente bem avaliados, gerando insumo acionável para estúdios,
publishers e analistas de mercado.
