from __future__ import annotations

import argparse
import gzip
import os
import sys
from typing import Iterable, List, Tuple

import numpy as np
import networkx as nx

try:
    from scipy.sparse import csr_matrix
except Exception:
    csr_matrix = None


def read_edge_list(path: str, directed: bool = True) -> Iterable[Tuple[str, str]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            yield u, v
            if not directed:
                yield v, u


def build_sparse_transition(edges: Iterable[Tuple[str, str]]):
    if csr_matrix is None:
        raise RuntimeError("scipy is required for sparse PageRank implementation")

    # Map node id to index
    idx = {}
    sources = []
    targets = []
    for u, v in edges:
        if u not in idx:
            idx[u] = len(idx)
        if v not in idx:
            idx[v] = len(idx)
        sources.append(idx[u])
        targets.append(idx[v])

    n = len(idx)
    if n == 0:
        raise RuntimeError("No nodes found in edge list")

    # compute outdegree per column (source)
    outdeg = np.zeros(n, dtype=float)
    for s in sources:
        outdeg[s] += 1.0

    # Build data for matrix: M[row=target, col=source] = 1/outdeg[source]
    data = []
    rows = []
    cols = []
    for s, t in zip(sources, targets):
        if outdeg[s] > 0:
            rows.append(t)
            cols.append(s)
            data.append(1.0 / outdeg[s])

    M = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    nodes = [None] * n
    for node, i in idx.items():
        nodes[i] = node
    return M, nodes


def pagerank_power_iteration(M: "csr_matrix", damping: float = 0.85, tol: float = 1e-4, max_iter: int = 100) -> np.ndarray:
    if csr_matrix is None:
        raise RuntimeError("scipy is required for this function")

    n = M.shape[0]
    # initial vector
    pr = np.ones(n, dtype=float) / n

    # identify dangling columns (columns that sum to 0)
    col_sums = np.array(M.sum(axis=0)).ravel()
    dangling_mask = (col_sums == 0)

    teleport = (1.0 - damping) / n

    for iteration in range(1, max_iter + 1):
        prev = pr.copy()

        # contribution from linked nodes
        link_contrib = damping * (M.dot(prev))

        # dangling contribution: sum of PR at dangling nodes distributed uniformly
        dangling_sum = damping * prev[dangling_mask].sum() / n

        pr = link_contrib + dangling_sum + teleport

        # convergence check (L1)
        diff = np.abs(pr - prev).sum()
        if diff < tol:
            # normalize and return
            pr = pr / pr.sum()
            # print progress
            return pr

    # final normalize
    pr = pr / pr.sum()
    return pr


def load_edges_for_dataset(name: str, cache_dir: str = "data") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    if name == "cit-HepTh":
        # SNAP provides a gz file; use the standard URL
        url = "https://snap.stanford.edu/data/cit-HepTh.txt.gz"
        filename = os.path.join(cache_dir, "cit-HepTh.txt.gz")
    else:
        raise RuntimeError(f"Unknown dataset '{name}'")

    if not os.path.exists(filename):
        print(f"Downloading {name} -> {filename} ...")
        try:
            import urllib.request

            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")
    return filename


def compare_with_networkx(G: nx.DiGraph, pr_vector: np.ndarray, nodes: List[str], damping: float):
    """Compute networkx.pagerank and print a short comparison."""
    nx_pr = nx.pagerank(G, alpha=damping, weight="weight")
    # build vector in same order
    nx_vec = np.array([nx_pr.get(node, 0.0) for node in nodes], dtype=float)
    # normalize both
    nx_vec = nx_vec / nx_vec.sum()
    pr_vector = pr_vector / pr_vector.sum()
    # L1 difference
    diff = np.abs(pr_vector - nx_vec).sum()
    print(f"L1 difference between custom PageRank and networkx.pagerank (d={damping}): {diff:.6f}")
    return nx_vec


def build_nx_graph_from_edges_file(path: str, directed: bool = True) -> nx.DiGraph:
    G = nx.DiGraph() if directed else nx.Graph()
    for u, v in read_edge_list(path, directed=directed):
        G.add_edge(u, v)
    return G


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="PageRank from-scratch assignment")
    parser.add_argument("--edges", type=str, help="Path to edge list file (whitespace separated src dst). If omitted and --dataset provided, will download dataset.")
    parser.add_argument("--dataset", type=str, choices=["cit-HepTh"], help="Named dataset to download/use (optional)")
    parser.add_argument("--directed", action="store_true", default=True, help="Treat graph as directed (default)")
    parser.add_argument("--undirected", action="store_true", help="Treat graph as undirected (adds bidirectional edges)")
    parser.add_argument("--dampings", type=float, nargs="*", default=[0.85], help="Damping factors to evaluate (e.g. 0.5 0.85 0.99)")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance (L1) for power iteration")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations for power iteration")
    parser.add_argument("--top", type=int, default=10, help="Top-K nodes to show")
    parser.add_argument("--cache-dir", type=str, default="data", help="Cache dir for downloads")
    args = parser.parse_args(argv)

    if args.dataset and not args.edges:
        path = load_edges_for_dataset(args.dataset, cache_dir=args.cache_dir)
    elif args.edges:
        path = args.edges
    else:
        parser.error("Either --edges or --dataset must be provided")

    directed = not args.undirected and args.directed

    print(f"Loading edges from: {path} (directed={directed})")

    # Build sparse transition matrix and node list
    print("Building sparse transition matrix...")
    M, nodes = build_sparse_transition(read_edge_list(path, directed=directed))

    # Build networkx graph for comparison and for mapping to node ids
    print("Building NetworkX graph (for comparison)...")
    G = build_nx_graph_from_edges_file(path, directed=directed)

    for d in args.dampings:
        print("\n--- Damping factor: {} ---".format(d))
        pr_vec = pagerank_power_iteration(M, damping=d, tol=args.tol, max_iter=args.max_iter)

        # top-K from custom PR
        order = np.argsort(-pr_vec)
        print(f"Top {args.top} nodes (custom PageRank):")
        for rank, idx in enumerate(order[: args.top], start=1):
            print(f"{rank:2d}. {nodes[idx]}  score={pr_vec[idx]:.6e}")

        # compare with networkx
        try:
            nx_vec = compare_with_networkx(G, pr_vec, nodes, d)
            order_nx = np.argsort(-nx_vec)
            print(f"Top {args.top} nodes (networkx.pagerank):")
            for rank, idx in enumerate(order_nx[: args.top], start=1):
                print(f"{rank:2d}. {nodes[idx]}  score={nx_vec[idx]:.6e}")
        except Exception as e:
            print(f"networkx comparison failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
