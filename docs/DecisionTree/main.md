## 1	Exploração dos Dados 20

``` python exec="on" html="0"
--8<-- "./docs/DecisionTree/limpa-dados.py"
```

## 2	Pré-processamento 10

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

``` python exec="on" html="0"
--8<-- "./docs/DecisionTree/arvore-treino.py"
```

<object type="image/svg+xml" data="arvore.svg"></object>

O teste com 80% atingiu 99% de acurácia, ou seja, overfitting.

## Teste com 70% dos dados para treino

``` python exec="on" html="0"
--8<-- "./docs/DecisionTree/arvore-treino2.py"
```

O teste com 70% atingiu 99% de acurácia, ou seja, overfitting.

## Teste com 60% dos dados para treino

``` python exec="on" html="0"
--8<-- "./docs/DecisionTree/arvore-treino3.py"
```

O teste com 60% atingiu 99% de acurácia, ou seja, overfitting.

## Teste com 50% dos dados para treino

``` python exec="on" html="0"
--8<-- "./docs/DecisionTree/arvore-treino4.py"
```

O teste com 50% atingiu 99% de acurácia, ou seja, overfitting.

## 5	Avaliação do Modelo	20

Com os testes realizados, é possível ver que com esta base de dados é impossível fazer um modelo confiável com esta técnica. Talvez, usando alguma outra técnica ou com mais dados, seria possível.


## 6	Relatório Final

Neste projeto, foi realizada a análise e modelagem de dados utilizando a técnica de árvore de decisão. O processo envolveu a exploração dos dados, pré-processamento, codificação de variáveis categóricas, divisão dos dados em conjuntos de treino e teste, e avaliação do desempenho do modelo.

Os testes realizados com diferentes proporções de dados para treino (80%, 70%, 60% e 50%) mostraram que o modelo atingiu alta acurácia (99%), indicando ocorrência de overfitting. Isso significa que o modelo está ajustado demais aos dados de treino e pode não generalizar bem para novos dados.

A partir dos resultados, conclui-se que, com esta base de dados e técnica utilizada, não é possível construir um modelo confiável. Recomenda-se testar outras técnicas de machine learning ou utilizar uma base de dados maior e mais variada para obter resultados mais robustos.
