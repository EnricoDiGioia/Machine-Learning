import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans

plt.figure(figsize=(12, 10))



# Carregar dados do CSV do link
url = "https://raw.githubusercontent.com/EnricoDiGioia/Machine-Learning/refs/heads/main/data/audi.csv"
df = pd.read_csv(url)

# Selecionar duas colunas numéricas para o K-means
# Exemplo usando 'year' e 'price'
X = df[["mileage", "price"]].values


# Run K-Means
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)

# Adicionar os rótulos ao DataFrame
df['cluster'] = labels

# Mapear cores do cmap para clusters
import matplotlib as mpl
cmap = plt.get_cmap('viridis')
colors = [cmap(i / (kmeans.n_clusters - 1)) for i in range(kmeans.n_clusters)]

# Plotar cada cluster com legenda do tipo de transmissão mais comum
for cluster in range(kmeans.n_clusters):
    cluster_data = df[df['cluster'] == cluster]
    # Tipo de transmissão mais comum no cluster
    if not cluster_data.empty:
        common_trans = cluster_data['transmission'].mode()[0]
        plt.scatter(cluster_data['mileage'], cluster_data['price'], 
                    color=colors[cluster], s=50, 
                    label=f'Cluster {cluster} ({common_trans})')

# Plotar centróides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='*', s=200, label='Centroids')
plt.title('K-Means Clustering Results')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.legend()

# Adicionar os rótulos ao DataFrame
#df['cluster'] = labels

# Verificar a relação entre cluster e transmissão
#print(pd.crosstab(df['cluster'], df['transmission']))

# # Print centroids and inertia
# print("Final centroids:", kmeans.cluster_centers_)
# print("Inertia (WCSS):", kmeans.inertia_)

# # Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())