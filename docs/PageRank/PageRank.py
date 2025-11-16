"""
PageRank sobre grafo de similaridade dos anúncios em `audi.csv`.

Abordagem:
- Cada anúncio (linha do CSV) é um nó.
- Conectamos cada nó aos seus `k` vizinhos mais próximos
  (baseado em features numéricas normalizadas) criando um grafo dirigido.
- Executamos PageRank no grafo para ordenar anúncios por centralidade.

Uso:
python PageRank.py --url <CSV_URL> --k 10 --top 20

Saída:
- `pagerank_results.csv` com as colunas originais mais `pagerank_rank` e `pagerank_score`.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import networkx as nx


def build_similarity_graph(X: np.ndarray, k: int = 10) -> nx.DiGraph:
	n = X.shape[0]
	# Para eficiência, usamos sklearn NearestNeighbors quando disponível
	try:
		from sklearn.neighbors import NearestNeighbors

		nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto").fit(X)
		distances, indices = nbrs.kneighbors(X)
	except Exception:
		# Fallback ingênuo (pode ser lento em datasets grandes)
		dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
		indices = np.argsort(dists, axis=1)[:, : min(k + 1, n)]
		distances = np.take_along_axis(dists, indices, axis=1)

	G = nx.DiGraph()
	G.add_nodes_from(range(n))

	for i in range(n):
		neighs = indices[i]
		dists_i = distances[i]
		# primeiro vizinho é o próprio ponto (dist ~ 0) — descartamos
		for j, dist in zip(neighs[1:], dists_i[1:]):
			weight = 1.0 / (1.0 + float(dist))
			G.add_edge(i, int(j), weight=weight)

	return G


def compute_pagerank(df: pd.DataFrame, feature_cols: List[str], k: int = 10, alpha: float = 0.85) -> pd.DataFrame:
	df2 = df.copy()

	# Extrai features numéricas, converte e preenche NA com mediana
	X = df2[feature_cols].apply(pd.to_numeric, errors="coerce")
	X = X.fillna(X.median())

	# Normaliza (z-score)
	Xn = (X - X.mean()) / (X.std().replace(0, 1))
	Xarr = Xn.values.astype(float)

	G = build_similarity_graph(Xarr, k=k)

	# PageRank usando pesos
	pr = nx.pagerank(G, alpha=alpha, weight="weight")

	# mapear scores de volta ao DataFrame
	scores = np.array([pr.get(i, 0.0) for i in range(len(df2))])
	df2["pagerank_score"] = scores
	df2 = df2.sort_values("pagerank_score", ascending=False).reset_index(drop=True)
	df2["pagerank_rank"] = df2["pagerank_score"].rank(method="first", ascending=False).astype(int)
	return df2


def main():
	parser = argparse.ArgumentParser(description="PageRank via grafo de similaridade (audi.csv)")
	parser.add_argument("--url", type=str, default=(
		"https://raw.githubusercontent.com/EnricoDiGioia/Machine-Learning/main/data/audi.csv"
	), help="URL do CSV (padrão: audi.csv no repo fornecido)")
	parser.add_argument("--k", type=int, default=10, help="Número de vizinhos por nó")
	parser.add_argument("--top", type=int, default=20, help="Quantos resultados top mostrar")
	parser.add_argument("--out", type=str, default="pagerank_results.csv", help="Arquivo de saída CSV")
	args = parser.parse_args()

	print(f"Lendo CSV de: {args.url}")
	df = pd.read_csv(args.url)

	# limpeza básica
	df.columns = df.columns.str.strip()
	if "model" in df.columns:
		df["model"] = df["model"].astype(str).str.strip()

	# escolher colunas numéricas relevantes
	candidate_numeric = ["year", "price", "mileage", "tax", "mpg", "engineSize"]
	feature_cols = [c for c in candidate_numeric if c in df.columns]
	if not feature_cols:
		raise RuntimeError("Nenhuma coluna numérica encontrada para construir features.")

	print(f"Usando features: {feature_cols}")

	result = compute_pagerank(df, feature_cols, k=args.k)

	# salvar resultados
	out_path = os.path.join(os.path.dirname(__file__), args.out)
	result.to_csv(out_path, index=False)
	print(f"Resultados salvos em: {out_path}")

	# mostrar top-N
	topn = result.head(args.top)
	print("\nTop {0} por PageRank:".format(min(args.top, len(result))))
	print(topn[[col for col in ["model", "year", "price", "pagerank_score"] if col in topn.columns]].to_string(index=False))


if __name__ == "__main__":
	main()

