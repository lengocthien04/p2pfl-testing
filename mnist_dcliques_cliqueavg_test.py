import torch
torch.set_num_threads(1)

import time
import argparse
import os
from datetime import datetime
from collections import Counter
from typing import Dict, Any

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.examples.mnist.model.mlp_pytorch import MLP
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy
from p2pfl.node import Node
from p2pfl.utils.d_cliques_p2pfl import build_dcliques_adjacency_matrix
from p2pfl.learning.aggregators.d_sgd_clique_avg import DSGDCliqueAvg


def connect_from_matrix(matrix: list[list[int]], nodes: list[Node]) -> None:
    """Connect nodes according to an undirected adjacency matrix."""
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] == 1:
                nodes[i].connect(nodes[j].addr)
                time.sleep(0.05)


def extract_label_counts_from_shard(shard: Any, label_key: str = "label") -> Dict[str, int]:
    """Extract label distribution from a data shard."""
    if not hasattr(shard, "_data"):
        raise RuntimeError(f"Expected P2PFLDataset with _data, got: {type(shard)}")

    data = shard._data

    if hasattr(shard, "_train_split_name"):
        train_name = shard._train_split_name
        if isinstance(data, dict) and train_name in data:
            ds = data[train_name]
        elif hasattr(data, "__getitem__"):
            try:
                ds = data[train_name]
            except Exception:
                ds = data
        else:
            ds = data
    else:
        ds = data

    if hasattr(ds, "column_names") and label_key in getattr(ds, "column_names", []):
        labels = ds[label_key]
        return dict(Counter(str(int(y)) for y in labels))

    if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
        c = Counter()
        for i in range(len(ds)):
            item = ds[i]
            if isinstance(item, dict) and label_key in item:
                c[str(int(item[label_key]))] += 1
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                c[str(int(item[1]))] += 1
            else:
                raise RuntimeError(f"Unknown item structure: {type(item)} at i={i}")
        return dict(c)

    raise RuntimeError(f"Unsupported inner ds type: {type(ds)}")


def get_clique_assignment(node_order: list[str], clique_size: int) -> Dict[str, set]:
    """
    Determine which clique each node belongs to based on node_order.
    
    Returns:
        Dict mapping node_id to set of clique members (including itself)
    """
    clique_map = {}
    num_cliques = (len(node_order) + clique_size - 1) // clique_size
    
    for clique_idx in range(num_cliques):
        start_idx = clique_idx * clique_size
        end_idx = min(start_idx + clique_size, len(node_order))
        clique_members = set(node_order[start_idx:end_idx])
        
        for node_id in clique_members:
            clique_map[node_id] = clique_members
    
    return clique_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--base-port", type=int, default=6666)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--clique-size", type=int, default=5)
    ap.add_argument("--swap-iters", type=int, default=500)
    ap.add_argument("--inter-mode", type=str, default="small_world",
                    choices=["small_world", "ring", "fractal", "fully_connected"])
    ap.add_argument("--small-world-c", type=int, default=2)
    ap.add_argument("--clique-weight-ratio", type=float, default=2.0,
                    help="Weight ratio for clique members vs inter-clique neighbors")

    args = ap.parse_args()
    
    # Configure Ray actor pool
    from p2pfl.settings import Settings
    Settings.training.RAY_ACTOR_POOL_SIZE = min(args.n, 10)
    
    # Enable neighbor-only aggregation for TRUE D-SGD
    Settings.training.NEIGHBOR_ONLY_AGGREGATION = True
    print(f"✅ Neighbor-only aggregation ENABLED (true D-SGD)")
    
    Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS = 50
    print(f"Ray actor pool size set to {Settings.training.RAY_ACTOR_POOL_SIZE}")

    # Load dataset + partition
    dataset = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    dataset.set_batch_size(128)
    strategy = DirichletPartitionStrategy()

    datashards = dataset.generate_partitions(
        num_partitions=args.n,
        strategy=strategy,
        seed=args.seed,
        label_tag="label",
        alpha=args.alpha,
    )

    # Build node_ids matching node list order
    node_order = [f"node_{i}" for i in range(args.n)]

    # Compute node_labels from datashards
    node_labels: Dict[str, Dict[str, int]] = {}
    for i in range(args.n):
        node_id = node_order[i]
        node_labels[node_id] = extract_label_counts_from_shard(datashards[i])

    print("\n=== Label Distribution (Dirichlet alpha={}) ===".format(args.alpha))
    for node_id, counts in node_labels.items():
        print(f"{node_id}: {counts}")
    print("="*50 + "\n")

    # Build D-Cliques adjacency matrix
    matrix = build_dcliques_adjacency_matrix(
        node_labels=node_labels,
        node_order=node_order,
        clique_size=args.clique_size,
        iterations=args.swap_iters,
        seed=args.seed,
        inter_mode=args.inter_mode,
        small_world_c=args.small_world_c,
    )
    
    # Get clique assignments
    clique_map = get_clique_assignment(node_order, args.clique_size)
    
    print("\n=== D-Cliques Topology with Clique Averaging ===")
    for i, node_id in enumerate(node_order):
        neighbors = [node_order[j] for j in range(len(node_order)) if matrix[i][j] == 1]
        clique_members = clique_map[node_id]
        print(f"{node_id}: {len(neighbors)} neighbors, clique: {clique_members}")
    print("="*50 + "\n")

    # Create nodes with D-SGD Clique Averaging aggregator
    nodes: list[Node] = []
    aggregators = []
    
    for i in range(args.n):
        node_id = node_order[i]
        aggregator = DSGDCliqueAvg(clique_weight_ratio=args.clique_weight_ratio)
        
        # Convert node_ids to addresses for clique members
        clique_addrs = {f"127.0.0.1:{args.base_port + node_order.index(nid)}" 
                       for nid in clique_map[node_id]}
        aggregator.set_clique_members(clique_addrs)
        
        node = Node(
            model=LightningModel(MLP(lr_rate=0.1)),
            data=datashards[i],
            addr=f"127.0.0.1:{args.base_port + i}",
            aggregator=aggregator,
        )
        node.start()
        nodes.append(node)
        aggregators.append(aggregator)

    # Connect according to D-Cliques matrix
    connect_from_matrix(matrix, nodes)

    # Start learning
    nodes[0].set_start_learning(rounds=args.rounds, epochs=args.epochs)

    # Wait until all nodes finish
    while True:
        time.sleep(1)
        if all(n.state.round is None for n in nodes):
            break

    # Export communication logs
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("logs", "comm", f"mnist_dcliques_cliqueavg_{run_id}")
    os.makedirs(base_dir, exist_ok=True)

    for node in nodes:
        proto = getattr(node, "_communication_protocol", None)
        comm_logger = getattr(proto, "comm_logger", None) if proto else None
        if comm_logger is None:
            print(f"[WARN] No comm_logger for {node.addr}")
            continue
        fname = f"node_{node.addr.replace(':','_')}.csv"
        comm_logger.export_csv(os.path.join(base_dir, fname))

    print(f"\n✅ Communication logs saved to: {base_dir}")

    # Stop nodes
    for node in nodes:
        node.stop()

    print("✅ All nodes stopped. Training complete!")


if __name__ == "__main__":
    main()
