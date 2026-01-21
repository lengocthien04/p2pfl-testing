import time
import argparse

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.examples.mnist.model.mlp_pytorch import MLP
from p2pfl.node import Node

# Your TopologyFactory / TopologyType are in topologies.py
# Adjust this import path to wherever you placed that file.
from p2pfl.utils.topologies import TopologyFactory, TopologyType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--base-port", type=int, default=6666)
    ap.add_argument("--topology", type=str, default="random_3")  # random_2 | random_3 | random_4 | ring | full | ...
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    # 1) Create & start N nodes
    nodes = []
    for i in range(args.n):
        node = Node(
            model=LightningModel(MLP()),
            data=P2PFLDataset.from_huggingface("p2pfl/MNIST"),
            addr=f"127.0.0.1:{args.base_port + i}",
        )
        node.start()
        nodes.append(node)

    # 2) Generate random topology matrix + connect nodes
    matrix = TopologyFactory.generate_matrix(args.topology, args.n)
    TopologyFactory.connect_nodes(matrix, nodes)

    # 3) Give connections a moment to stabilize
    time.sleep(3)

    # 4) Start learning from node 0
    # (broadcasts to the network + starts local learning thread)
    nodes[0].set_start_learning(rounds=args.rounds, epochs=args.epochs)  # :contentReference[oaicite:1]{index=1}

    # 5) Wait until learning finishes
    # node.state.round reads from experiment state :contentReference[oaicite:2]{index=2}
    while True:
        time.sleep(1)
        if nodes[0].state.round is None:
            break

    # 6) Stop all nodes
    for node in nodes:
        node.stop()  # :contentReference[oaicite:3]{index=3}


if __name__ == "__main__":
    main()
