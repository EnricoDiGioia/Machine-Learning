## 1	Exploração dos Dados 20

Nessa fase, buscamos compreender a estrutura, as características e possíveis inconsistências do conjunto de dados. Inicialmente, realizamos a leitura do dataset e a visualização das primeiras linhas para identificar o tipo de variáveis presentes, como colunas numéricas e categóricas, além de verificar a existência de valores ausentes ou discrepantes. Essa análise inicial permite entender melhor o comportamento dos dados, orientar a seleção de features relevantes e definir os próximos passos para a preparação e modelagem dos dados.

``` python exec="on" html="0"
--8<-- "./docs/DecisionTree/limpa-dados.py"
```

## 2	Pré-processamento 10

Para iniciar o processo de análise, foi realizado o carregamento do dataset de carros Audi, disponível em formato CSV. Em seguida, foi feita uma amostragem aleatória de 10 registros para facilitar a visualização e manipulação inicial dos dados.

Na etapa de pré-processamento, valores ausentes nas colunas numéricas (tax, mpg e price) foram preenchidos com a mediana de cada coluna, garantindo que não houvesse dados faltantes que pudessem prejudicar as análises subsequentes. Após o tratamento dos dados, foram selecionadas as principais variáveis de interesse: model, year, price, transmission, mileage, fuelType, tax, mpg e engineSize.

A visualização das primeiras linhas do dataset permitiu verificar a estrutura dos dados, identificar possíveis inconsistências e compreender melhor as variáveis disponíveis para a construção dos modelos de machine learning.

2.1 Data cleaning

``` python exec="on" html="0"
--8<-- "./docs/DecisionTree/arvore-limpa.py"
```

## 3 Treinamento do modelo

Com os dados devidamente preparados e divididos, foi realizado o treinamento do modelo KNN (K-Nearest Neighbors). O modelo foi ajustado utilizando o conjunto de treino, onde o algoritmo aprende a identificar padrões e relações entre as variáveis de entrada (features) e o alvo (rótulo).

Durante o treinamento, o KNN armazena os exemplos do conjunto de treino e, ao receber uma nova amostra, classifica-a com base nos vizinhos mais próximos, de acordo com a métrica de distância escolhida. O parâmetro k define o número de vizinhos considerados para a decisão. Esse processo é fundamental para que o modelo possa realizar previsões precisas em novos dados.

=== "Code"

    ``` python 
    --8<-- "./docs/KNN/KNN01.py"
    ```

## 4 Avaliação do Modelo	20

Após o treinamento do modelo KNN, a avaliação de desempenho foi realizada utilizando o conjunto de teste separado anteriormente. Para isso, o modelo fez previsões sobre os dados de teste e a acurácia foi calculada por meio da função accuracy_score da biblioteca scikit-learn.

A acurácia representa a proporção de previsões corretas em relação ao total de exemplos avaliados, sendo uma métrica simples e direta para problemas de classificação. Esse processo permite verificar se o modelo está generalizando bem para dados que não foram vistos durante o treinamento, fornecendo uma estimativa confiável de seu desempenho em situações reais.
=== "Code"

    ``` python 
    --8<-- "./docs/KNN/KNN01.py"
    ```


``` python exec="on" html="1"
--8<-- "./docs/KNN/KNN01.py"
```

## 5	Relatório Final

O processo iniciou-se com a exploração dos dados, onde foram identificadas as principais características do conjunto e tratados eventuais valores ausentes. Em seguida, os dados foram divididos em conjuntos de treino e teste, garantindo uma avaliação imparcial do modelo.

O treinamento do modelo KNN foi realizado com o conjunto de treino, utilizando as variáveis mais relevantes para a tarefa de classificação. Após o ajuste, o modelo foi avaliado com o conjunto de teste, sendo a acurácia utilizada como principal métrica de desempenho. O resultado obtido demonstrou a capacidade do modelo em generalizar para novos dados, validando a abordagem adotada.

Por fim, a visualização da fronteira de decisão permitiu compreender como o modelo separa as diferentes classes no espaço das features escolhidas. O experimento evidenciou a importância das etapas de preparação dos dados, escolha adequada das variáveis e validação do modelo para o sucesso em projetos de machine learning.