# cifar10_resnet_full.py
import time
import argparse
import os
from datetime import datetime
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy
from p2pfl.node import Node

from p2pfl.utils.topologies import TopologyFactory

# ---- import the ResNet build function you pasted ----
# Adjust this import path to match where the file actually is in YOUR repo.
# Example if you saved it as: p2pfl/examples/cifar10/model/resnet_cifar10.py
from p2pfl.examples.cifar10.model.resnet_pytorch import model_build_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--base-port", type=int, default=6666)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--trainset-size", type=int, default=4)
    ap.add_argument("--experiment-name", type=str, default="cifar10-resnet-full")
    args = ap.parse_args()

    # ---- load CIFAR10 dataset ----
    # [Unverified] dataset id string. If this fails, find the correct one in your repo.
    dataset = P2PFLDataset.from_huggingface("p2pfl/CIFAR10")
    dataset.batch_size = 20

    # ---- partition (Dirichlet alpha) ----
    strategy = DirichletPartitionStrategy()
    datashards = dataset.generate_partitions(
        num_partitions=args.n,
        strategy=strategy,
        seed=666,
        label_tag="label",
        alpha=args.alpha,
    )

    # ---- create & start nodes ----
    nodes: list[Node] = []
    for i in range(args.n):
        node = Node(
            model=model_build_fn(),   # ResNet CIFAR10 LightningModel wrapper
            data=datashards[i],
            addr=f"127.0.0.1:{args.base_port + i}",
            # aggregator=YourDSGD()  # <- only if you already implemented DSGD as an Aggregator subclass
        )
        node.start()
        nodes.append(node)

    # ---- FULLY CONNECTED topology ----
    matrix = TopologyFactory.generate_matrix("full", args.n)
    TopologyFactory.connect_nodes(matrix, nodes)

    # ---- start learning from node 0 ----
    nodes[0].set_start_learning(
        rounds=args.rounds,
        epochs=args.epochs,
        trainset_size=args.trainset_size,
        experiment_name=args.experiment_name,
    )

    # ---- wait until finished ----
    while True:
        time.sleep(1)
        if nodes[0].state.round is None:
            break
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("logs", "comm", f"run_{run_id}")
    os.makedirs(base_dir, exist_ok=True)

    for node in nodes:
        proto = getattr(node, "protocol", None)
        comm_logger = getattr(proto, "comm_logger", None) if proto else None
        if comm_logger is not None:
            fname = f"cifar10_fully_node_{node.addr.replace(':','_')}.csv"
            comm_logger.export_csv(os.path.join(base_dir, fname))

    # ---- stop nodes ----
    for node in nodes:
        node.stop()


if __name__ == "__main__":
    main()
