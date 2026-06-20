### Instruções

# Análise de Dados e Solução de Problemas com Python

## 1. Objetivo do Trabalho

Neste projeto, vocês atuarão como Cientista de Dados para resolver um problema real de sua escolha. O objetivo é demonstrar o domínio do ecossistema Python ensinado ao longo da disciplina (Numpy, Pandas, Matplotlib e Scikit-learn) aplicando-o em um ciclo completo de dados: desde a coleta e limpeza dos dados até a geração de insights visuais e a aplicação de um modelo preditivo ou descritivo(A plicação de um modelo é opcional).

## 2. O que deve ser feito?

Vocês têm total liberdade para escolher a base de dados e a pergunta de negócio/problema a ser resolvido. Sugere-se buscar dados abertos em portais como Kaggle. O tema pode variar desde a análise de logs de performance de um sistema de software, até dados financeiros, saúde ou automotivos.

Independentemente do tema, o trabalho deve, obrigatoriamente, conter as seguintes etapas:

1. **Definição do Problema:** Qual problema do mundo real vocês estão tentando resolver ou entender com esses dados?
2. **Obtenção e Preparação dos Dados:** Carregamento do dataset e uso de Pandas/Numpy para tratar dados nulos, remover duplicatas, alterar tipos e criar novas features necessárias.
3. **Análise Exploratória e Visualização:** Criação de uma interface streamlit, que contenha gráficos explicativos utilizando o Matplotlib para entender a distribuição dos dados e encontrar padrões.
4. **Conclusão (Relatório):** Um resumo executivo explicando os achados, as limitações do modelo(caso tenha feito um modelo de ML) e como isso resolve a pergunta inicial.

## 3. Entregáveis

- **Código-Fonte:** Um ou mais arquivos `.py` e os dados base utilizados(Tem que estar em anexo no ava, não somente o link do git). O código deve rodar de ponta a ponta sem erros.
- **Relatório Breve:** Pode ser entregue como um PDF. Deve conter: _Introdução ao Problema, Metodologia (o que foi feito com os dados), Resultados Visuais, Avaliação do Modelo(Caso tenha feito um Modelo) e Conclusão._

---

## 4. Métricas de Avaliação (Valor Total: 2,5 Pontos)

A nota final será calculada de 0 a 2,5, distribuída nas seguintes métricas objetivas:

**Critério 1: Definição do Problema, Obtenção e Tratamento de Dados (1,3 Pontos)**

- **(1,3)** O problema de negócio está bem formulado e faz sentido para o _dataset_ escolhido. Houve uso correto e robusto de **Pandas** e **Numpy**, com esforço real e justificado na limpeza (nulos/duplicatas), conversão de tipos e criação de novas _features_ (engenharia de dados) essenciais para resolver a pergunta proposta.
- **(0,65)** O problema é vago, o _dataset_ é simplório demais ou o tratamento foi superficial. Fez o básico de leitura, mas ignorou dados inconsistentes ou não realizou as transformações exigidas pelo contexto do problema.
- **(0,0)** Não há uma formulação clara do problema e/ou nenhuma manipulação relevante; os dados foram passados para a interface em seu estado bruto original.

**Critério 2: Construção da Interface e Visualização de Dados (0,8 Ponto)**

- **(0,8)** A aplicação em **Streamlit** está funcional e bem estruturada. Os gráficos gerados (com **Matplotlib**) são claros, possuem eixos/legendas e geram _insights_ valiosos que respondem ao problema proposto. A usabilidade da interface agrega valor à análise.
- **(0,4)** A interface executa, mas é confusa, trava, ou os gráficos estão mal formatados, não ajudando a explicar os dados ou o problema.
- **(0,0)** Não entregou a interface em Streamlit ou não gerou nenhuma visualização útil.

**Critério 3: Qualidade do Código e Estrutura (0,2 Pontos)**

- **(0,2)** Código limpo, bem estruturado e legível, sem repetições excessivas de blocos lógicos. Apropriado para o nível de alunos de Engenharia de Software.
- **(0,1)** Código funcional, mas desorganizado, com variáveis mal nomeadas ou sem nenhuma modularidade.
- **(0,0)** O código sequer executa de ponta a ponta (erros impeditivos na inicialização do Streamlit).

**Critério 4: Relatório Auxiliar de Entrega (0,2 Pontos)**

- **(0,2)** O relatório cumpre bem o seu papel de registro documental: apresenta rapidamente o escopo, detalha a metodologia de tratamento e consolida as conclusões de forma coerente.
- **(0,1)** Relatório genérico, feito apenas para constar, com prints soltos e sem ligar os dados à conclusão final.
- **(0,0)** Não anexou o relatório em PDF.

---