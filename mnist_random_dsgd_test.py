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
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.learning.aggregators.d_sgd import DSGD


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="Number of nodes")
    ap.add_argument("--base-port", type=int, default=6666)
    ap.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    ap.add_argument("--epochs", type=int, default=1, help="Epochs per round")
    ap.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha for non-IID data")
    ap.add_argument("--seed", type=int, default=666, help="Random seed")
    ap.add_argument("--avg-degree", type=int, default=4, choices=[2, 3, 4, 5], 
                    help="Average degree for random topology (2, 3, 4, or 5)")

    args = ap.parse_args()

    # Configure Ray actor pool
    from p2pfl.settings import Settings
    Settings.training.RAY_ACTOR_POOL_SIZE = min(args.n, 10)
    print(f"Ray actor pool size set to {Settings.training.RAY_ACTOR_POOL_SIZE}")

    # Load and partition MNIST dataset
    dataset = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitioner = DirichletPartitionStrategy(alpha=args.alpha, seed=args.seed)
    datashards = partitioner.create_partitions(dataset, args.n)

    # Print label distribution
    print("\n=== Label Distribution (Dirichlet alpha={}) ===".format(args.alpha))
    for i, shard in enumerate(datashards):
        counts = extract_label_counts_from_shard(shard)
        print(f"Node {i}: {counts}")
    print("="*50 + "\n")

    # Generate random topology
    topology_map = {2: TopologyType.RANDOM_2, 3: TopologyType.RANDOM_3, 4: TopologyType.RANDOM_4, 5: TopologyType.RANDOM_5}
    topology_type = topology_map[args.avg_degree]
    
    matrix = TopologyFactory.generate_matrix(topology_type, args.n)
    
    # Print topology info
    print(f"\n=== Random Topology (avg degree {args.avg_degree}) ===")
    for i in range(args.n):
        neighbors = [j for j in range(args.n) if matrix[i][j] == 1]
        print(f"Node {i}: {len(neighbors)} neighbors -> {neighbors}")
    print("="*50 + "\n")

    # Create nodes with D-SGD aggregator
    nodes: list[Node] = []
    for i in range(args.n):
        node = Node(
            model=LightningModel(MLP(lr_rate=0.002)),
            data=datashards[i],
            addr=f"127.0.0.1:{args.base_port + i}",
            aggregator=DSGD(),
        )
        node.start()
        nodes.append(node)

    # Connect nodes according to topology
    TopologyFactory.connect_nodes(matrix, nodes)

    # Start learning
    nodes[0].set_start_learning(rounds=args.rounds, epochs=args.epochs)

    # Wait until all nodes finish
    while True:
        time.sleep(1)
        if all(n.state.round is None for n in nodes):
            break

    # Export communication logs
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("logs", "comm", f"mnist_random_deg{args.avg_degree}_{run_id}")
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
