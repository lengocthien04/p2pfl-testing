import torch
torch.set_num_threads(1)  # CRITICAL: Limit PyTorch threads to avoid CPU oversubscription

import argparse
import time
import os
from datetime import datetime
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.examples.mnist.model.mlp_pytorch import MLP
from p2pfl.node import Node

from p2pfl.utils.topologies import TopologyFactory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--base-port", type=int, default=6666)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=666)
    args = ap.parse_args()

    # 1) Dataset + Dirichlet partition
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

    # 2) Create & start nodes
    nodes: list[Node] = []
    for i in range(args.n):
        node = Node(
            model=LightningModel(MLP(lr_rate=0.1)),
            data=datashards[i],
            addr=f"127.0.0.1:{args.base_port + i}",
            # If you implemented DSGD as an Aggregator:
            # aggregator=DSGD(),
            #
            # OR if p2pfl expects it as "learner/strategy/protocol",
            # keep default here and change inside your workflow config.
        )
        node.start()
        nodes.append(node)

    # 3) Fully-connected topology
    matrix = TopologyFactory.generate_matrix("full", args.n)
    TopologyFactory.connect_nodes(matrix, nodes)

    # 4) Start decentralized learning from node 0
    nodes[0].set_start_learning(rounds=args.rounds, epochs=args.epochs)

    # 5) Wait until finished
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
            fname = f"mnist_fully_node_{node.addr.replace(':','_')}.csv"
            comm_logger.export_csv(os.path.join(base_dir, fname))


    # 6) Stop nodes
    for node in nodes:
        node.stop()


if __name__ == "__main__":
    main()
