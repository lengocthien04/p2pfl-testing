# topologies/d_cliques_p2pfl.py
# D-Cliques-style topology builder for p2pfl:
# 1) Greedy-Swap (K iterations) to build cliques that reduce label-skew
# 2) inter(DC): ring / fractal / small_world (power-of-two offsets) / fully_connected over cliques
# 3) Assign inter-clique edges to specific node pairs by load-balanced greedy selection
#    (NO ring-star hub logic; bridge nodes are just endpoints of inter-clique edges)

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

LabelDist = Dict[str, float]


# ---------------------------
# Label distributions + skew
# ---------------------------
def _normalize(counts: Mapping[str, float]) -> LabelDist:
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("Label counts must sum to a positive value")
    return {label: value / total for label, value in counts.items()}


def compute_label_distribution(node_labels: Mapping[str, Mapping[str, float] | str]) -> Dict[str, LabelDist]:
    """
    Normalizes label distributions per node and returns node_id -> distribution.
    Accepts either:
      - a mapping label->count (e.g. {"0": 50, "1": 10})
      - a single label string (treated as one-hot count=1)
    """
    normalized: Dict[str, LabelDist] = {}
    for node_id, labels in node_labels.items():
        if isinstance(labels, str):
            counts = Counter({labels: 1.0})
        else:
            counts = Counter({k: float(v) for k, v in labels.items()})
        normalized[node_id] = _normalize(counts)
    return normalized


def _aggregate_clique_distribution(clique: Iterable[str], node_distributions: Mapping[str, LabelDist]) -> LabelDist:
    agg: Dict[str, float] = defaultdict(float)
    for node in clique:
        if node not in node_distributions:
            raise ValueError(f"Node '{node}' missing label distribution")
        for label, prob in node_distributions[node].items():
            agg[label] += prob
    return _normalize(agg)


def compute_skew(clique: Iterable[str], node_distributions: Mapping[str, LabelDist], global_distribution: LabelDist) -> float:
    """
    L1 distance between clique label distribution and global distribution.
    Lower is better (more balanced).
    """
    clique_dist = _aggregate_clique_distribution(clique, node_distributions)
    labels = set(clique_dist) | set(global_distribution)
    return sum(abs(clique_dist.get(label, 0.0) - global_distribution.get(label, 0.0)) for label in labels)


# ---------------------------
# 1) Greedy-Swap clique builder
# ---------------------------
def build_d_cliques(
    node_labels: Mapping[str, Mapping[str, float] | str],
    clique_size: int,
    iterations: int = 1000,         # K in "greedy swap for K steps"
    seed: int | None = None,
) -> List[Set[str]]:
    """
    D-Cliques-style greedy swap:
      - Start with random partition into cliques of size clique_size (last clique may be smaller)
      - For K iterations:
          pick 2 cliques, try swaps that reduce combined skew, perform one improving swap
    """
    if clique_size <= 0:
        raise ValueError("clique_size must be positive")

    nodes = list(node_labels.keys())
    if not nodes:
        raise ValueError("node_labels cannot be empty")

    rng = random.Random(seed)
    rng.shuffle(nodes)

    node_distributions = compute_label_distribution(node_labels)

    # Global distribution = normalized sum of per-node distributions
    global_counts: Counter[str] = Counter()
    for dist in node_distributions.values():
        for label, prob in dist.items():
            global_counts[label] += prob
    global_distribution = _normalize(global_counts)

    # Initial partition
    cliques: List[Set[str]] = []
    for i in range(0, len(nodes), clique_size):
        cliques.append(set(nodes[i : i + clique_size]))

    if len(cliques) <= 1:
        return cliques

    for _ in range(iterations):
        idx_a, idx_b = rng.sample(range(len(cliques)), 2)
        clique_a = cliques[idx_a]
        clique_b = cliques[idx_b]

        base_skew = compute_skew(clique_a, node_distributions, global_distribution) + compute_skew(
            clique_b, node_distributions, global_distribution
        )

        improvements: List[Tuple[str, str]] = []
        # Evaluate all cross-swaps for a strictly improving move
        for a in clique_a:
            for b in clique_b:
                new_a = set(clique_a)
                new_b = set(clique_b)
                new_a.remove(a); new_a.add(b)
                new_b.remove(b); new_b.add(a)

                new_skew = compute_skew(new_a, node_distributions, global_distribution) + compute_skew(
                    new_b, node_distributions, global_distribution
                )
                if new_skew < base_skew:
                    improvements.append((a, b))

        if improvements:
            swap_a, swap_b = rng.choice(improvements)
            cliques[idx_a].remove(swap_a); cliques[idx_a].add(swap_b)
            cliques[idx_b].remove(swap_b); cliques[idx_b].add(swap_a)

    return _merge_singleton_cliques(cliques, node_distributions, global_distribution)


def _merge_singleton_cliques(
    cliques: List[Set[str]],
    node_distributions: Mapping[str, LabelDist],
    global_distribution: LabelDist,
) -> List[Set[str]]:
    """
    Optional hygiene: avoid singleton cliques by merging into least-disruptive clique.
    This is not a core requirement of D-Cliques, but it prevents degenerate tiny cliques.
    """
    if len(cliques) <= 1:
        return cliques

    idx = 0
    while idx < len(cliques):
        clique = cliques[idx]
        if len(clique) > 1:
            idx += 1
            continue

        node = next(iter(clique))
        best_target: int | None = None
        best_delta = float("inf")

        for target_idx, target_clique in enumerate(cliques):
            if target_idx == idx:
                continue
            current_skew = compute_skew(target_clique, node_distributions, global_distribution)
            candidate = set(target_clique); candidate.add(node)
            new_skew = compute_skew(candidate, node_distributions, global_distribution)
            delta = new_skew - current_skew
            if delta < best_delta:
                best_delta = delta
                best_target = target_idx

        if best_target is None:
            idx += 1
            continue

        cliques[best_target].add(node)
        cliques.pop(idx)

        if len(cliques) <= 1:
            break

    return cliques


# ---------------------------
# 2) inter(DC): clique-graph edges
# ---------------------------
def _add_edge(edges: Set[Tuple[int, int]], a: int, b: int) -> None:
    if a == b:
        return
    edges.add((min(a, b), max(a, b)))


def build_interclique_edges_dcliques(
    num_cliques: int,
    mode: str = "small_world",
    small_world_c: int = 2,
) -> List[Tuple[int, int]]:
    """
    Build inter-clique edges (between clique indices) in D-Cliques spirit.

    mode:
      - "ring": connect i <-> i+1
      - "fractal": connect i <-> i + stride, where stride=max(2, num_cliques//2)
      - "small_world": power-of-two ring offsets: for k in [0..c-1], connect i <-> i±2^k
                      (this is the "exponentially bigger hop sets" pattern)
      - "fully_connected": complete graph over cliques

    Note: D-Cliques describes the exponential-hop small-world-like pattern at the CLIQUE level.
          Bridge nodes are just endpoints chosen later.
    """
    if num_cliques <= 1:
        return []
    if mode not in {"ring", "fractal", "small_world", "fully_connected"}:
        raise ValueError(f"Unknown mode '{mode}'")
    if mode == "small_world" and small_world_c <= 0:
        raise ValueError("small_world_c must be positive")

    edges: Set[Tuple[int, int]] = set()

    # Base ring connectivity (keeps graph connected)
    for i in range(num_cliques):
        _add_edge(edges, i, (i + 1) % num_cliques)

    if mode == "ring":
        return sorted(edges)

    if mode == "fully_connected":
        for i in range(num_cliques):
            for j in range(i + 1, num_cliques):
                _add_edge(edges, i, j)
        return sorted(edges)

    if mode == "fractal":
        stride = max(2, num_cliques // 2)
        for i in range(num_cliques):
            _add_edge(edges, i, (i + stride) % num_cliques)
        return sorted(edges)

    # small_world: power-of-two offsets BOTH directions
    for k in range(small_world_c):
        offset = 2**k
        for i in range(num_cliques):
            _add_edge(edges, i, (i + offset) % num_cliques)
            _add_edge(edges, i, (i - offset) % num_cliques)

    return sorted(edges)


# ---------------------------
# 3) Assign clique-edges to node-edges (load-balanced)
# ---------------------------
def _select_lowest_degree_node(clique_nodes: Set[str], node_edge_count: Mapping[str, int]) -> str:
    # deterministic tie-break by node id string
    return min(clique_nodes, key=lambda n: (node_edge_count[n], n))


def assign_node_edges_balanced(
    cliques: Sequence[Set[str]],
    interclique_edges: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    Convert clique-index edges into node-id edges by greedy load balancing:
      for each (clique_a, clique_b):
        pick node in clique_a with lowest current degree
        pick node in clique_b with lowest current degree
        connect them
    Initial degree counts include intra-clique full connections: (|clique|-1).
    """
    node_edge_count: Dict[str, int] = {}
    for clique in cliques:
        base = len(clique) - 1  # intra-clique degree in a complete clique
        for node in clique:
            node_edge_count[node] = base

    node_edges: List[Tuple[str, str]] = []

    for ca, cb in interclique_edges:
        a = _select_lowest_degree_node(cliques[ca], node_edge_count)
        b = _select_lowest_degree_node(cliques[cb], node_edge_count)
        node_edges.append((a, b))
        node_edge_count[a] += 1
        node_edge_count[b] += 1

    return node_edges, node_edge_count


# ---------------------------
# 4) Build adjacency matrix for p2pfl
# ---------------------------
def build_dcliques_adjacency_matrix(
    node_labels: Mapping[str, Mapping[str, float] | str],
    node_order: Sequence[str],
    clique_size: int,
    iterations: int = 1000,
    seed: int | None = None,
    inter_mode: str = "small_world",
    small_world_c: int = 2,
) -> List[List[int]]:
    """
    Returns adjacency matrix (0/1) in the order of node_order (must match p2pfl nodes list order).
    """
    # Build cliques using ONLY node ids found in node_order (and node_labels)
    missing = [n for n in node_order if n not in node_labels]
    if missing:
        raise ValueError(f"node_labels missing entries for: {missing}")

    # Restrict labels to node_order (so experiments match p2pfl node list)
    labels_restricted = {n: node_labels[n] for n in node_order}

    cliques = build_d_cliques(labels_restricted, clique_size, iterations, seed)

    # Clique-graph edges
    interclique = build_interclique_edges_dcliques(
        num_cliques=len(cliques),
        mode=inter_mode,
        small_world_c=small_world_c,
    )

    # Intra edges (complete within each clique)
    intra_edges: List[Tuple[str, str]] = []
    for clique in cliques:
        members = sorted(clique)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                intra_edges.append((members[i], members[j]))

    # Inter edges assigned to specific node pairs (balanced)
    inter_edges, _ = assign_node_edges_balanced(cliques, interclique)

    # Build adjacency matrix
    idx = {node_id: i for i, node_id in enumerate(node_order)}
    N = len(node_order)
    A = [[0] * N for _ in range(N)]

    def add(u: str, v: str) -> None:
        i, j = idx[u], idx[v]
        if i == j:
            return
        A[i][j] = 1
        A[j][i] = 1

    for u, v in intra_edges:
        add(u, v)
    for u, v in inter_edges:
        add(u, v)

    return A
