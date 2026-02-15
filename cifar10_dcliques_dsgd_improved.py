"""
Improved CIFAR-10 D-Cliques DSGD test with better error handling and monitoring.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""         
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
os.environ["OMP_NUM_THREADS"] = "1"  # Reduce thread contention

import time
import argparse
from datetime import datetime
from collections import Counter
from typing import Dict, Any

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.examples.cifar10.model.resnet_pytorch import ResNetCIFAR10
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy
from p2pfl.node import Node
from p2pfl.utils.d_cliques_p2pfl import build_dcliques_adjacency_matrix
from p2pfl.learning.aggregators.d_sgd import DSGD
from p2pfl.settings import Settings


def connect_from_matrix(matrix: list[list[int]], nodes: list[Node]) -> None:
    """Connect nodes according to an undirected adjacency matrix."""
    n = len(nodes)
    connections = 0
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] == 1:
                nodes[i].connect(nodes[j].addr)
                connections += 1
    print(f"✅ Created {connections} bidirectional connections")

               
def extract_label_counts_from_shard(shard: Any, label_key: str = "label") -> Dict[str, int]:
    """Extract label counts from P2PFLDataset shard."""
    if not hasattr(shard, "_data"):
        raise RuntimeError(f"Expected P2PFLDataset with _data, got: {type(shard)}")

    data = shard._data

    # Handle train split
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

    # Extract labels
    if hasattr(ds, "column_names") and label_key in getattr(ds, "column_names", []):
        labels = ds[label_key]
        return dict(Counter(str(int(y)) for y in labels))

    # Fallback
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
    ap = argparse.ArgumentParser(description="CIFAR-10 D-Cliques DSGD Training")
    ap.add_argument("--n", type=int, default=2, help="Number of nodes")
    ap.add_argument("--base-port", type=int, default=6666, help="Base port for nodes")
    ap.add_argument("--rounds", type=int, default=5, help="Training rounds")
    ap.add_argument("--epochs", type=int, default=1, help="Epochs per round")

    # Dirichlet partition
    ap.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha")
    ap.add_argument("--seed", type=int, default=666, help="Random seed")

    # D-Cliques params
    ap.add_argument("--clique-size", type=int, default=5, help="Clique size")
    ap.add_argument("--swap-iters", type=int, default=500, help="Swap iterations")
    ap.add_argument("--inter-mode", type=str, default="small_world",
                    choices=["small_world", "ring", "fractal", "fully_connected"],
                    help="Inter-clique connection mode")
    ap.add_argument("--small-world-c", type=int, default=2, help="Small world parameter")

    args = ap.parse_args()

    print("=" * 80)
    print("CIFAR-10 D-Cliques DSGD Training (Improved)")
    print("=" * 80)
    print(f"Nodes: {args.n}")
    print(f"Rounds: {args.rounds}")
    print(f"Epochs per round: {args.epochs}")
    print(f"Dirichlet alpha: {args.alpha}")
    print(f"Clique size: {args.clique_size}")
    print(f"Inter-clique mode: {args.inter_mode}")
    print("=" * 80)

    # Configure settings for better reliability
    # CRITICAL: HEARTBEAT_TIMEOUT must be longer than training time per round
    Settings.heartbeat.TIMEOUT = 300.0  # Prevent neighbor removal during training
    Settings.general.GRPC_TIMEOUT = 120.0  # Allow large model transmission
    Settings.training.AGGREGATION_TIMEOUT = 600  # Wait longer than heartbeat timeout
    
    # Enable neighbor-only aggregation for TRUE D-SGD
    Settings.training.NEIGHBOR_ONLY_AGGREGATION = True
    print(f"✅ Neighbor-only aggregation ENABLED (true D-SGD)")
    
    print(f"⚙️  Settings: HEARTBEAT_TIMEOUT={Settings.heartbeat.TIMEOUT}s, "
          f"GRPC_TIMEOUT={Settings.general.GRPC_TIMEOUT}s, "
          f"AGGREGATION_TIMEOUT={Settings.training.AGGREGATION_TIMEOUT}s")

    # 1) Load dataset + partition
    print("\n📊 Loading and partitioning CIFAR-10 dataset...")
    dataset = P2PFLDataset.from_huggingface("p2pfl/CIFAR10")
    dataset.batch_size = 20
    strategy = DirichletPartitionStrategy()

    datashards = dataset.generate_partitions(
        num_partitions=args.n,
        strategy=strategy,
        seed=args.seed,
        label_tag="label",
        alpha=args.alpha,
    )
    print(f"✅ Created {len(datashards)} data partitions")

    # 2) Build node_ids and extract labels
    node_order = [f"node_{i}" for i in range(args.n)]
    
    print("\n🏷️  Extracting label distributions...")
    node_labels: Dict[str, Dict[str, int]] = {}
    for i in range(args.n):
        node_id = node_order[i]
        node_labels[node_id] = extract_label_counts_from_shard(datashards[i])
    print(f"✅ Label distributions extracted")

    # 3) Build D-Cliques adjacency matrix
    print(f"\n🔗 Building D-Cliques topology...")
    matrix = build_dcliques_adjacency_matrix(
        node_labels=node_labels,
        node_order=node_order,
        clique_size=args.clique_size,
        iterations=args.swap_iters,
        seed=args.seed,
        inter_mode=args.inter_mode,
        small_world_c=args.small_world_c,
    )
    
    # Calculate average degree
    avg_degree = sum(sum(row) for row in matrix) / len(matrix)
    print(f"✅ Topology created. Average degree: {avg_degree:.2f}")

    # 4) Create nodes with DSGD aggregator
    print(f"\n🚀 Starting {args.n} nodes...")
    nodes: list[Node] = []
    for i in range(args.n):
        node = Node(
            model=LightningModel(ResNetCIFAR10(lr_rate=0.002)),
            data=datashards[i],
            addr=f"127.0.0.1:{args.base_port + i}",
            aggregator=DSGD(),
        )
        node.start()
        nodes.append(node)
        if (i + 1) % 10 == 0:
            print(f"  Started {i + 1}/{args.n} nodes...")
    print(f"✅ All {args.n} nodes started")

    # 5) Connect according to D-Cliques matrix
    print("\n🔗 Connecting nodes...")
    connect_from_matrix(matrix, nodes)
    
    print("⏳ Waiting for network stabilization (10s)...")
    time.sleep(10)

    # 6) Setup logging
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("logs", "comm", f"run_{run_id}")
    os.makedirs(base_dir, exist_ok=True)
    
    for node in nodes:
        fname = f"cifar10_dcliques_node_{node.addr.replace(':','_')}.csv"
        node.comm_logger.set_file_path(os.path.join(base_dir, fname))
    
    print(f"📝 Logs will be saved to: {base_dir}")

    # 7) Start learning
    print(f"\n🎓 Starting training: {args.rounds} rounds x {args.epochs} epochs")
    print("=" * 80)
    
    start_time = time.time()
    nodes[0].set_start_learning(rounds=args.rounds, epochs=args.epochs)

    # Monitor progress
    last_round_check = -1
    stall_count = 0
    last_status_check = ""
    
    while True:
        time.sleep(5)
        
        # Check if all nodes finished
        if all(n.state.round is None for n in nodes):
            print("\n✅ All nodes finished training!")
            break
        
        # Monitor progress
        current_rounds = [n.state.round for n in nodes if n.state.round is not None]
        current_status = [n.state.status for n in nodes if n.state.round is not None]
        
        if current_rounds:
            min_round = min(current_rounds)
            max_round = max(current_rounds)
            avg_round = sum(current_rounds) / len(current_rounds)
            
            # Create status summary
            status_summary = f"R{min_round}-{max_round}:{Counter(current_status)}"
            
            # Check if round changed OR status changed (e.g., evaluating → training)
            if min_round != last_round_check or status_summary != last_status_check:
                elapsed = time.time() - start_time
                status_str = ", ".join([f"{k}:{v}" for k, v in Counter(current_status).items()])
                print(f"⏱️  Round {min_round}-{max_round} (avg={avg_round:.1f}) | "
                      f"Status: {status_str} | Elapsed: {elapsed:.0f}s")
                last_round_check = min_round
                last_status_check = status_summary
                stall_count = 0
            else:
                stall_count += 1
                # Only warn if stuck for 5 minutes (not during evaluation which is slow)
                if stall_count > 60:  # 5 minutes with no progress
                    print(f"⚠️  WARNING: No progress for 5 minutes. Nodes may be stuck.")
                    print(f"   Current rounds: {Counter(current_rounds)}")
                    print(f"   Current status: {Counter(current_status)}")
                    # Reset counter to avoid spam
                    stall_count = 0

    # 8) Final save
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total training time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
    for node in nodes:
        if node.comm_logger:
            node.comm_logger.save()

    # 9) Stop nodes
    print("\n🛑 Stopping nodes...")
    for node in nodes:
        node.stop()
    
    print("✅ Training complete!")
    print(f"📊 Check logs at: {base_dir}")


if __name__ == "__main__":
    main()
