import torch
torch.set_num_threads(1)  # CRITICAL: Must be before any other imports!

import time
import argparse
import os
from datetime import datetime

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.examples.cifar10.model.resnet_pytorch import ResNet9
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy
from p2pfl.node import Node
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.learning.aggregators.d_sgd import DSGD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="Number of nodes")
    ap.add_argument("--base-port", type=int, default=6666)
    ap.add_argument("--rounds", type=int, default=50, help="Number of training rounds")
    ap.add_argument("--epochs", type=int, default=1, help="Epochs per round")
    ap.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha for non-IID data")
    ap.add_argument("--seed", type=int, default=666, help="Random seed")
    ap.add_argument("--avg-degree", type=int, default=4, choices=[2, 3, 4, 5], 
                    help="Average degree for random topology (2, 3, 4, or 5)")
    ap.add_argument("--batch-size", type=int, default=20, help="Batch size for training")

    args = ap.parse_args()

    # Configure Ray actor pool
    from p2pfl.settings import Settings
    Settings.training.RAY_ACTOR_POOL_SIZE = min(args.n, 10)
    print(f"Ray actor pool size set to {Settings.training.RAY_ACTOR_POOL_SIZE}")
    
    # CRITICAL: Enable neighbor-only aggregation for true D-SGD
    Settings.training.NEIGHBOR_ONLY_AGGREGATION = True
    print(f"✅ Neighbor-only aggregation ENABLED (true D-SGD)")
    
    # Configure timeouts for large networks
    Settings.heartbeat.TIMEOUT = 300.0  # 5 minutes for heartbeat
    Settings.general.GRPC_TIMEOUT = 60.0  # 1 minute for GRPC
    Settings.training.AGGREGATION_TIMEOUT = 1800  # 30 minutes - allow time for all nodes to train
    
    # Increase gossip exit threshold for sparse topology
    # Random topology needs more time for models to propagate through multiple hops
    Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS = 100  # Much higher to prevent early exit
    
    print(f"⚙️  Configured timeouts:")
    print(f"   Heartbeat timeout: {Settings.heartbeat.TIMEOUT}s")
    print(f"   GRPC timeout: {Settings.general.GRPC_TIMEOUT}s")
    print(f"   Aggregation timeout: {Settings.training.AGGREGATION_TIMEOUT}s")
    print(f"   Gossip exit threshold: {Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS} equal rounds")

    # Load and partition CIFAR-10 dataset
    dataset = P2PFLDataset.from_huggingface("p2pfl/CIFAR10")
    dataset.set_batch_size(20)
    
    strategy = DirichletPartitionStrategy()
    datashards = dataset.generate_partitions(
        num_partitions=args.n,
        strategy=strategy,
        seed=args.seed,
        label_tag="label",
        alpha=args.alpha,
    )

    print(f"\n=== CIFAR-10 Dataset ===")
    print(f"Nodes: {args.n}")
    print(f"Alpha: {args.alpha} (Dirichlet non-IID)")
    print(f"Batch size: 20")
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
            model=LightningModel(ResNet9(lr_rate=0.002)),
            data=datashards[i],
            addr=f"127.0.0.1:{args.base_port + i}",
            aggregator=DSGD(),
        )
        node.start()
        nodes.append(node)

    # Connect nodes according to topology
    TopologyFactory.connect_nodes(matrix, nodes)

    # Start learning with all nodes in trainset
    # Gossip mechanism creates temporary connections to reach all trainset nodes
    print(f"\n⚙️  Starting learning with trainset_size={args.n} (all nodes)")
    nodes[0].set_start_learning(rounds=args.rounds, epochs=args.epochs, trainset_size=args.n)

    # Wait until all nodes finish
    while True:
        time.sleep(1)
        if all(n.state.round is None for n in nodes):
            break

    # Export communication logs
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("logs", "comm", f"cifar10_random_deg{args.avg_degree}_{run_id}")
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
