
## 1. Carregamento dos Dados
Os dados utilizados são provenientes de um arquivo CSV hospedado online, contendo informações sobre carros, como quilometragem (`mileage`), preço (`price`) e tipo de transmissão (`transmission`). O carregamento é feito com pandas:

```python
import pandas as pd
url = "https://raw.githubusercontent.com/EnricoDiGioia/Machine-Learning/refs/heads/main/data/audi.csv"
df = pd.read_csv(url)
```

## 2. Seleção de Variáveis para Clustering
Para o agrupamento, são escolhidas duas variáveis numéricas: `mileage` e `price`. Elas são extraídas do DataFrame para formar a matriz de entrada do K-Means:

```python
X = df[["mileage", "price"]].values
```

## 3. Execução do K-Means
O algoritmo K-Means é aplicado para dividir os dados em grupos (clusters) com base na similaridade dessas variáveis.

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)
df['cluster'] = labels
```

## 4. Visualização dos Resultados
Os clusters são visualizados em um gráfico de dispersão, onde cada cor representa um cluster. A legenda mostra, para cada cluster, a distribuição dos tipos de transmissão mais comuns:

```python
import matplotlib.pyplot as plt
cmap = plt.get_cmap('viridis')
colors = [cmap(i / (kmeans.n_clusters - 1)) for i in range(kmeans.n_clusters)]
legend_labels = []
for cluster in range(kmeans.n_clusters):
	cluster_data = df[df['cluster'] == cluster]
	trans_counts = cluster_data['transmission'].value_counts(normalize=True).head(2)
	parts = [f"{t} {p*100:.0f}%" for t, p in trans_counts.items()]
	legend_labels.append(f"Cluster {cluster}: " + " | ".join(parts))
	plt.scatter(cluster_data['mileage'], cluster_data['price'], color=colors[cluster], s=50)
for i, txt in enumerate(legend_labels):
	plt.scatter([], [], color=colors[i], label=txt)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='*', s=200, label='Centroids')
plt.legend()
plt.title('K-Means Clustering Results')
plt.xlabel('Mileage')
plt.ylabel('Price')
```

## 5. Interpretação dos Clusters
Após o agrupamento, é possível analisar a relação entre os clusters e o tipo de transmissão. O K-Means não utiliza `transmission` para formar os grupos, mas a distribuição desse atributo em cada cluster pode ser observada na legenda e em tabelas cruzadas:

```python
print(pd.crosstab(df['cluster'], df['transmission']))
```

## 6. Limitações e Considerações
- O K-Means só considera as variáveis numéricas escolhidas.
- Se um tipo de transmissão for dominante, pode aparecer repetido na legenda de vários clusters.
- Para incluir variáveis categóricas no agrupamento, é necessário codificá-las (one-hot encoding) ou usar algoritmos específicos para dados mistos (ex.: k-prototypes).

## 7. Exemplos de Código
Os códigos trazem exemplos completos do processo descrito acima:

``` python exec="on" html="1"
--8<-- "./docs/K-Means/Kmeans.py"
```

O serguinte arquivo implementa o processo de agrupamento K-Means utilizando os dados de carros, considerando as variáveis `mileage` (quilometragem) e `price` (preço). Após o agrupamento, o script analisa a distribuição dos tipos de transmissão (`transmission`) dentro de cada cluster, exibindo essa informação na legenda do gráfico. Isso permite visualizar não apenas os grupos formados por características numéricas, mas também como o atributo categórico transmissão está distribuído entre os clusters, facilitando a interpretação dos resultados.

``` python exec="on" html="1"
--8<-- "./docs/K-Means/Kmeans4.py"
```