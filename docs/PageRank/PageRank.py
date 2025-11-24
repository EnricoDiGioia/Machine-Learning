import math
from typing import Dict, List


def pagerank(graph: Dict[str, List[str]], d: float = 0.85, tol: float = 1e-6, max_iter: int = 100) -> Dict[str, float]:
	"""
	Calcula o PageRank para um grafo direcionado representado como dicionário
	{node: [lista_de_nos_para_os_quais_node_aponta]}.

	Parâmetros:
	- graph: dicionário do grafo
	- d: damping factor (padrão 0.85)
	- tol: tolerância de convergência (critério L1)
	- max_iter: número máximo de iterações

	Retorna:
	- dict {node: pagerank_value}
	"""

	# Garantir que o conjunto de nós inclua nós apontados mas não declarados
	nodes = set(graph.keys())
	for targets in graph.values():
		for t in targets:
			nodes.add(t)

	nodes = sorted(nodes)
	n = len(nodes)

	# Construir lista de in-links e grau de saída
	in_links = {node: [] for node in nodes}
	out_degree = {node: 0 for node in nodes}

	for src, targets in graph.items():
		out_degree[src] = len(targets)
		for t in targets:
			in_links.setdefault(t, []).append(src)

	# Tratamento de nodes que não aparecem como chaves (dangling): out_degree = 0
	for node in nodes:
		out_degree.setdefault(node, 0)

	# Inicialização: todos com 1/n
	pr = {node: 1.0 / n for node in nodes}

	for iteration in range(1, max_iter + 1):
		new_pr = {}
		# soma de PageRank de nós dangling (sem saída)
		dangling_sum = sum(pr[node] for node in nodes if out_degree[node] == 0)

		for node in nodes:
			# teletransporte
			new_val = (1.0 - d) / n

			# contribuição de dangling nodes (distribuída igualmente)
			new_val += d * dangling_sum / n

			# somatório das contribuições dos in-links
			contrib = 0.0
			for src in in_links.get(node, []):
				if out_degree[src] > 0:
					contrib += pr[src] / out_degree[src]
			new_val += d * contrib

			new_pr[node] = new_val

		# Critério de convergência (norma L1)
		diff = sum(abs(new_pr[node] - pr[node]) for node in nodes)

		pr = new_pr

		if diff < tol:
			# Convergiu
			break

	# Normalizar (por segurança numérica)
	total = sum(pr.values())
	if not math.isclose(total, 1.0):
		pr = {node: val / total for node, val in pr.items()}

	return pr


if __name__ == "__main__":
	# Exemplo baseado no grafo fornecido pelo usuário
	graph = {
	    "C": ["D", "A"],
        "A": ["C"],
        "D": ["C", "B"],
        "B": ["A"]
	}

	print("Calculando PageRank (implementação própria)...")
	pr = pagerank(graph, d=0.85, tol=1e-6, max_iter=100)

	# Print sozinho: apenas o resultado da implementação própria
	print("PageRank (implementação própria) - Print sozinho:")
	for node, score in sorted(pr.items()):
		print(f"{node}: {round(score, 6)}")

	# Comparação com NetworkX (se disponível)
	try:
		import networkx as nx

		G = nx.DiGraph()
		for src, targets in graph.items():
			for t in targets:
				G.add_edge(src, t)

		nx_pr = nx.pagerank(G, alpha=0.85)

		# Print de comparação lado-a-lado
		print("\nComparação (implementação própria vs NetworkX):")
		print(f"{'Node':<6} {'OurPR':>10} {'NX_PR':>10} {'Diff':>10}")
		for node in sorted(set(list(pr.keys()) + list(nx_pr.keys()))):
			our = pr.get(node, 0.0)
			nxv = nx_pr.get(node, 0.0)
			diff = abs(our - nxv)
			print(f"{node:<6} {our:10.6f} {nxv:10.6f} {diff:10.6f}")
	except ImportError:
		print("\nNetworkX não está instalado; comparação lado-a-lado pulada.")

