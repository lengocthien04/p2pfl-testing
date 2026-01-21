# mnist_dsgd_random10.py
import time
import argparse

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.examples.mnist.model.mlp_pytorch import MLP
#There are many other data partition strat you can check
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy
from p2pfl.node import Node

# Use your topology utilities (random_2 | random_3 | random_4 | full | ring | ...)
# Can define more for future 
from p2pfl.utils.topologies import TopologyFactory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--base-port", type=int, default=6666)
    ap.add_argument("--topology", type=str, default="random_3")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    # Create & start N nodes
    nodes: list[Node] = []
    dataset = P2PFLDataset.from_huggingface("p2pfl/MNIST", )
    strategy = DirichletPartitionStrategy()
    # Data partition
    datashards = dataset.generate_partitions(
    num_partitions=10,
    strategy=strategy,
    seed=666,
    label_tag="label",
    alpha=0.1,         
    # min_partition_size=2,   # optional (exists in strategy)
    # self_balancing=False,   # optional
    )
    for i in range(args.n):
        node = Node(
            model=LightningModel(MLP()),
            data=datashards[i],
            addr=f"127.0.0.1:{args.base_port + i}",
            #By default
            #protocol= FedAvg
        )
        node.start()
        nodes.append(node)

    # Generate topology + connect nodes (neighbor-only aggregation is defined by this graph)
    matrix = TopologyFactory.generate_matrix(args.topology, args.n)
    TopologyFactory.connect_nodes(matrix, nodes)


    # start decentralized learning from node 0
    nodes[0].set_start_learning(rounds=args.rounds, epochs=args.epochs)

    # Wait until learning finishes
    while True:
        time.sleep(1)
        if nodes[0].state.round is None:
            break

    # Stop all nodes
    for node in nodes:
        node.stop()


if __name__ == "__main__":
    main()
