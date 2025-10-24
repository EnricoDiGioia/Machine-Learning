
# Random Forest para Classificação de Modelos

## Sobre o Script
O script abaixo utiliza o dataset `audi.csv`, que contém informações sobre carros da marca Audi, para prever o modelo do carro (`model`) a partir de características como ano, preço, transmissão, quilometragem, tipo de combustível, imposto, consumo (mpg) e tamanho do motor.

### Passos principais:
1. **Carregamento dos dados:** O dataset é lido diretamente de uma URL.
2. **Pré-processamento:**
	- Remoção de espaços em branco dos nomes das colunas.
	- Separação das variáveis preditoras (X) e do alvo (y).
	- Codificação das variáveis categóricas e do alvo usando `LabelEncoder`.
3. **Divisão dos dados:** Os dados são divididos em treino e teste (80%/20%).
4. **Treinamento:** Um modelo `RandomForestClassifier` é treinado para prever o modelo do carro.
5. **Avaliação:** O script imprime a acurácia do modelo e a importância de cada feature para a classificação.


## Código
``` python exec="on" html="1"
--8<-- "./docs/RandomFlorest/RadomFlorest.py"
```

## Interpretação da Saída

Ao executar o script, você verá uma saída semelhante a:

```
Accuracy: 0.5524835988753515
Feature Importances: [0.04144809 0.19694073 0.01491391 0.04202797 0.09726528 0.11844603 0.20736773 0.28159025]
```

### O que significa cada parte?

- **Accuracy:**
	- Representa a proporção de acertos do modelo no conjunto de teste. No exemplo acima, o modelo acertou cerca de 55% das previsões.
	- Em problemas de classificação com múltiplas classes (vários modelos Audi), valores em torno de 0.5 podem indicar que o modelo está aprendendo padrões, mas ainda há espaço para melhorias (por exemplo, com mais ajustes ou outros algoritmos).

- **Feature Importances:**
	- É um vetor que mostra a importância relativa de cada feature (coluna) para a decisão do modelo.
	- Quanto maior o valor, mais relevante aquela feature foi para a classificação.
	- A ordem dos valores corresponde à ordem das colunas em `X` após o pré-processamento:
		1. year
		2. price
		3. transmission
		4. mileage
		5. fuelType
		6. tax
		7. mpg
		8. engineSize
	- Por exemplo, no resultado acima, `engineSize` (0.28), `mpg` (0.20) e `price` (0.19) foram as variáveis mais importantes para prever o modelo do carro.

### Resumindo
O modelo Random Forest consegue identificar quais características dos carros Audi mais influenciam na distinção entre os diferentes modelos, além de fornecer uma métrica quantitativa de desempenho (acurácia) para avaliar sua performance.