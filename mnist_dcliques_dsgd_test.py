import torch
torch.set_num_threads(1)  # CRITICAL: Must be before any other imports!

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

from p2pfl.learning.aggregators.d_sgd import DSGD  # <-- change if needed


def connect_from_matrix(matrix: list[list[int]], nodes: list[Node]) -> None:
    """Connect nodes according to an undirected adjacency matrix."""
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] == 1:
                nodes[i].connect(nodes[j].addr)
                time.sleep(0.05)


def extract_label_counts_from_shard(shard: Any, label_key: str = "label") -> Dict[str, int]:
    """
    P2PFLDataset shard: labels live in shard._data.
    shard._data is typically a HuggingFace datasets.DatasetDict or Dataset.
    We try common patterns: dict-like splits, then Dataset["label"].
    """
    if not hasattr(shard, "_data"):
        raise RuntimeError(f"Expected P2PFLDataset with _data, got: {type(shard)}")

    data = shard._data  # <- confirmed by your debug

    # If it's a DatasetDict-like object: choose train split
    # P2PFLDataset stores split names too, so use them if present.
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

    # Now ds should behave like HuggingFace Dataset (supports ds["label"])
    if hasattr(ds, "column_names") and label_key in getattr(ds, "column_names", []):
        labels = ds[label_key]
        return dict(Counter(str(int(y)) for y in labels))

    # Fallback: try direct indexing dict-like items
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

    raise RuntimeError(
        f"Unsupported inner ds type: {type(ds)}. "
        f"Try printing type(shard._data) and type(ds)."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--base-port", type=int, default=6666)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1)

    # Dirichlet partition
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=666)

    # D-Cliques params
    ap.add_argument("--clique-size", type=int, default=5)
    ap.add_argument("--swap-iters", type=int, default=500)
    ap.add_argument("--inter-mode", type=str, default="small_world",
                    choices=["small_world", "ring", "fractal", "fully_connected"])
    ap.add_argument("--small-world-c", type=int, default=2)

    args = ap.parse_args()
    
    # CRITICAL: Configure Ray actor pool to match number of nodes
    from p2pfl.settings import Settings
    Settings.training.RAY_ACTOR_POOL_SIZE = min(args.n, 10)
    
    # Enable neighbor-only aggregation for TRUE D-SGD
    Settings.training.NEIGHBOR_ONLY_AGGREGATION = True
    print(f"✅ Neighbor-only aggregation ENABLED (true D-SGD)")
    
    # Configure timeouts for large networks
    Settings.heartbeat.TIMEOUT = 300.0  # 5 minutes for heartbeat
    Settings.general.GRPC_TIMEOUT = 60.0  # 1 minute for GRPC
    Settings.training.AGGREGATION_TIMEOUT = 1800  # 30 minutes
    
    # Increase gossip exit threshold for D-Cliques topology
    # D-Cliques needs more hops for messages to propagate
    Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS = 100
    
    print(f"Ray actor pool size set to {Settings.training.RAY_ACTOR_POOL_SIZE}")
    print(f"⚙️  Configured timeouts:")
    print(f"   Heartbeat timeout: {Settings.heartbeat.TIMEOUT}s")
    print(f"   GRPC timeout: {Settings.general.GRPC_TIMEOUT}s")
    print(f"   Aggregation timeout: {Settings.training.AGGREGATION_TIMEOUT}s")
    print(f"   Gossip exit threshold: {Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS} equal rounds")

    # 1) Load dataset + partition
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

    # 2) Build node_ids matching node list order
    node_order = [f"node_{i}" for i in range(args.n)]

    # 3) Compute node_labels (counts per label) from datashards
    print("Shard type:", type(datashards[0]))
    if hasattr(datashards[0], "column_names"):
       print("column_names:", datashards[0].column_names)
    sh = datashards[0]
    print("Shard type:", type(sh))
    print("Shard dict keys:", list(getattr(sh, "__dict__", {}).keys()))

    node_labels: Dict[str, Dict[str, int]] = {}
    for i in range(args.n):
        node_id = node_order[i]
        node_labels[node_id] = extract_label_counts_from_shard(datashards[i])

    # 4) Build D-Cliques adjacency matrix
    matrix = build_dcliques_adjacency_matrix(
        node_labels=node_labels,
        node_order=node_order,
        clique_size=args.clique_size,
        iterations=args.swap_iters,
        seed=args.seed,
        inter_mode=args.inter_mode,
        small_world_c=args.small_world_c,
    )
    
    # DEBUG: Print the adjacency matrix to verify topology
    print("\n=== D-Cliques Adjacency Matrix ===")
    print("Node order:", node_order)
    for i, node_id in enumerate(node_order):
        neighbors = [node_order[j] for j in range(len(node_order)) if matrix[i][j] == 1]
        print(f"{node_id}: {len(neighbors)} neighbors -> {neighbors}")
    print("="*50 + "\n")

    # 5) Create nodes (D-SGD aggregator)
    nodes: list[Node] = []
    for i in range(args.n):
        node = Node(
            model=LightningModel(MLP(lr_rate=0.1)),
            data=datashards[i],
            addr=f"127.0.0.1:{args.base_port + i}",
            aggregator=DSGD(),   # <-- key change vs FedAvg
        )
        node.start()
        nodes.append(node)

    # 6) Connect according to D-Cliques matrix
    connect_from_matrix(matrix, nodes)

   # 7) Start learning with small trainset (neighbor-only aggregation)
    trainset_size = min(10, args.n)  # Max 10 nodes in trainset
    print(f"\n⚙️  Starting learning with trainset_size={trainset_size}")
    print(f"   (Neighbor-only aggregation: nodes aggregate only from direct neighbors)")
    nodes[0].set_start_learning(rounds=args.rounds, epochs=args.epochs, trainset_size=trainset_size)

    # Wait until ALL nodes finished
    while True:
        time.sleep(1)
        if all(n.state.round is None for n in nodes):
            break

    # 8) Export comm logs
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("logs", "comm", f"run_{run_id}")
    os.makedirs(base_dir, exist_ok=True)

    for node in nodes:
        proto = getattr(node, "_communication_protocol", None)
        comm_logger = getattr(proto, "comm_logger", None) if proto else None
        if comm_logger is None:
            print(f"[WARN] No comm_logger for {node.addr}")
            continue
        if comm_logger is not None:
            fname = f"mnist_dcliques_node_{node.addr.replace(':','_')}.csv"
            comm_logger.export_csv(os.path.join(base_dir, fname))

    # 9) Stop nodes
    for node in nodes:
        node.stop()


if __name__ == "__main__":
    main()
