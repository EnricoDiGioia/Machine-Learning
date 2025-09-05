## 1	Exploração dos Dados 20

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

2.2 Data Encoding Categorical variables

``` python exec="on" html="0"
--8<-- "./docs/DecisionTree/arvore-categoria.py"
```

3	Divisão dos Dados 20

4	Treinamento do Modelo 10

## Teste com 80% dos dados para treino

Após a etapa de exploração e pré-processamento dos dados, foi realizada a divisão do dataset em conjuntos de treino e teste. Essa separação é fundamental para avaliar o desempenho do modelo de machine learning de forma imparcial, garantindo que o algoritmo seja treinado em uma parte dos dados e validado em outra, nunca vista durante o treinamento.

Utilizou-se a função train_test_split da biblioteca scikit-learn, que permite dividir os dados de forma aleatória, mantendo a proporção definida entre treino e teste (por exemplo, 80% para treino e 20% para teste). Dessa forma, é possível medir a capacidade de generalização do modelo e evitar problemas como overfitting.

=== "Code"

    ``` python 
    --8<-- "./docs/DecisionTree/arvore-treino.py"
    ```
=== "Dados"

    ``` python exec="on" html="1"
    --8<-- "docs/DecisionTree/arvore-treino.py"
    ```

O teste com 80% atingiu 99% de acurácia, ou seja, overfitting.

## Teste com 70% dos dados para treino

``` python exec="on" html="1"
--8<-- "./docs/DecisionTree/arvore-treino2.py"
```


O teste com 70% atingiu 99% de acurácia, ou seja, overfitting.

## Teste com 60% dos dados para treino

``` python exec="on" html="1"
--8<-- "./docs/DecisionTree/arvore-treino3.py"
```

O teste com 60% atingiu 98% de acurácia, ou seja, overfitting.

## Teste com 50% dos dados para treino

``` python exec="on" html="1"
--8<-- "./docs/DecisionTree/arvore-treino4.py"
```

O teste com 50% atingiu 98% de acurácia, ou seja, overfitting.

## 5	Avaliação do Modelo	20

Com os testes realizados, é possível ver que com esta base de dados é impossível fazer um modelo confiável com esta técnica. Talvez, usando alguma outra técnica ou com mais dados, seria possível.


## 6	Relatório Final

Neste projeto, foi realizada a análise e modelagem de dados utilizando a técnica de árvore de decisão. O processo envolveu a exploração dos dados, pré-processamento, codificação de variáveis categóricas, divisão dos dados em conjuntos de treino e teste, e avaliação do desempenho do modelo.

Os testes realizados com diferentes proporções de dados para treino (80%, 70%, 60% e 50%) mostraram que o modelo atingiu alta acurácia (99%), indicando ocorrência de overfitting. Isso significa que o modelo está ajustado demais aos dados de treino e pode não generalizar bem para novos dados.

A partir dos resultados, conclui-se que, com esta base de dados e técnica utilizada, não é possível construir um modelo confiável. Recomenda-se testar outras técnicas de machine learning ou utilizar uma base de dados maior e mais variada para obter resultados mais robustos.
