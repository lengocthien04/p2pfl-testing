#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""Launch from YAMLs."""

import importlib
import os
import time
import uuid
from typing import Any

import yaml

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory
from p2pfl.utils.utils import wait_convergence, wait_to_finish


def load_by_package_and_name(package_name, class_name) -> Any:
    """
    Load a class by package and name.

    Args:
        package_name: The package name.
        class_name: The class name.

    """
    module = importlib.import_module(package_name)
    return getattr(module, class_name)


def run_from_yaml(yaml_path: str, debug: bool = False) -> None:
    """
    Run a simulation from a YAML file.

    Args:
        yaml_path: The path to the YAML file.
        debug: If True, enable debug mode.

    """
    # Parse YAML configuration
    with open(yaml_path) as file:
        config = yaml.safe_load(file)

    # Update settings
    custom_settings = config.get("settings", {})
    if custom_settings:
        Settings.set_from_dict(custom_settings)
        # Refresh (already initialized)
        logger.set_level(Settings.general.LOG_LEVEL)

    # Get Amount of Nodes
    network_config = config.get("network", {})
    if not network_config:
        raise ValueError("Missing 'network' configuration in YAML file.")
    n = network_config.get("nodes")
    if not n:
        raise ValueError("Missing 'n' under 'network' configuration in YAML file.")

    #############
    # Profiling #
    #############

    profiling = config.get("profiling", {})
    profiling_enabled = profiling.get("enabled", False)
    profiling_output_dir = profiling.get("output_dir", "profile")
    if profiling_enabled:
        import yappi  # type: ignore

        # Start profiler
        yappi.start()

    start_time = None
    if profiling.get("measure_time", False):
        start_time = time.time()

    ###################
    # Remote Loggers  #
    ###################

    remote_loggers = config.get("remote_loggers", {})
    if remote_loggers:
        logger.connect(**remote_loggers)

    ###########
    # Dataset #
    ###########

    experiment_config = config.get("experiment", {})
    dataset_config = experiment_config.get("dataset", {})  # Get dataset config
    if not dataset_config:
        raise ValueError("Missing 'dataset' configuration in YAML file.")
    data_source = dataset_config.get("source")
    if not data_source:
        raise ValueError("Missing 'source' under 'dataset' configuration in YAML file.")
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise ValueError("Dataset source is 'huggingface' but 'name' is missing in YAML.")

    # Load data
    dataset = None
    if data_source == "huggingface":
        dataset = P2PFLDataset.from_huggingface(dataset_name)
    elif data_source == "csv":
        dataset = P2PFLDataset.from_csv(dataset_name)
    elif data_source == "json":
        dataset = P2PFLDataset.from_json(dataset_name)
    elif data_source == "parquet":
        dataset = P2PFLDataset.from_parquet(dataset_name)
    elif data_source == "pandas":
        dataset = P2PFLDataset.from_pandas(dataset_name)
    elif data_source == "custom":
        # Get custom dataset configuration
        package = dataset_config.get("package")
        dataset_class = dataset_config.get("class")
        if not package or not dataset_class:
            raise ValueError("Missing package or class for custom dataset")

        # Load custom dataset class
        dataset_class = load_by_package_and_name(package, dataset_class)
        dataset = dataset_class(**dataset_config.get("params", {}))

    if not dataset:
        print("P2PFLDataset loading process completed without creating a dataset object (check for errors above).")
        return None

    # Batch size
    dataset.set_batch_size(dataset_config.get("batch_size", 1))
    # Optional DataLoader knobs (PyTorch). Keep generic so they can be consumed by export strategies.
    dataloader_keys = ("num_workers", "pin_memory", "persistent_workers", "prefetch_factor")
    dataloader_kwargs = {k: dataset_config[k] for k in dataloader_keys if k in dataset_config}
    if dataloader_kwargs:
        dataset.set_dataloader_kwargs(dataloader_kwargs)

    # Partitioning (do this BEFORE applying transforms)
    partitioning_config = dataset_config.get("partitioning", {})
    if not partitioning_config:
        raise ValueError("Missing 'partitioning' configuration in YAML file.")
    partition_package = partitioning_config.get("package")
    partition_class_name = partitioning_config.get("strategy")
    if not partition_package or not partition_class_name:
        raise ValueError("Missing 'partition_strategy' configuration in YAML file.")
    reduced_dataset = partitioning_config.get("reduced_dataset", False)
    reduction_factor = partitioning_config.get("reduction_factor", 1)
    partitions = dataset.generate_partitions(
        n * reduction_factor if reduced_dataset else n,
        load_by_package_and_name(
            partition_package,
            partition_class_name,
        ),
        **partitioning_config.get("params", {}),
    )

    # Transforms (apply AFTER partitioning)
    transforms_config = dataset_config.get("transforms", None)
    if transforms_config:
        transforms_package = transforms_config.get("package")
        transform_function = transforms_config.get("function")
        if not transforms_package or not transform_function:
            raise ValueError("Missing 'transforms' configuration in YAML file.")
        transform_class = load_by_package_and_name(
            transforms_package,
            transform_function,
        )
        # Apply transforms to each partition
        for partition in partitions:
            partition.set_transforms(transform_class(**transforms_config.get("params", {})))

    #########
    # Model #
    #########

    model_config = experiment_config.get("model", {})
    if not model_config:
        raise ValueError("Missing 'model' configuration in YAML file.")
    model_package = model_config.get("package")
    model_build_fn = model_config.get("model_build_fn")
    if not model_package or not model_build_fn:
        raise ValueError("Missing 'model' configuration in YAML file.")
    model_class = load_by_package_and_name(
        model_package,
        model_build_fn,
    )

    def model_fn() -> P2PFLModel:
        params = model_config.get("params", {})
        params = {**params, "compression": model_config.get("compression", None)}
        return model_class(**params)

    ##############
    # Aggregator #
    ##############

    aggregator = experiment_config.get("aggregator")
    if not aggregator:
        raise ValueError("Missing 'aggregator' configuration in YAML file.")
    aggregator_package = aggregator.get("package")
    aggregator_class_name = aggregator.get("aggregator")
    if not aggregator_package or not aggregator_class_name:
        raise ValueError("Missing 'aggregator' configuration in YAML file.")
    aggregator_class = load_by_package_and_name(
        aggregator_package,
        aggregator_class_name,
    )

    def aggregator_fn() -> Aggregator:
        return aggregator_class(**aggregator.get("params", {}))

    ###########
    # Network #
    ###########

    # Create nodes
    nodes: list[Node] = []
    protocol_package = network_config.get("package")
    protocol_class_name = network_config.get("protocol")
    if not protocol_package or not protocol_class_name:
        raise ValueError("Missing 'protocol' configuration in YAML file.")
    protocol = load_by_package_and_name(
        protocol_package,
        protocol_class_name,
    )
    
    # Setup comm logger directory
    import os
    from datetime import datetime
    experiment_name = experiment_config.get("name", "experiment")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    comm_log_dir = os.path.join("logs", "comm", f"{experiment_name}_{run_id}")
    os.makedirs(comm_log_dir, exist_ok=True)
    
    for i in range(n):
        node = Node(
            model_fn(),
            partitions[i],
            protocol=protocol(),
            aggregator=aggregator_fn(),
        )
        node.start()
        
        # Setup auto-save for comm logger
        node_name = f"node_{i}"
        comm_log_path = os.path.join(comm_log_dir, f"{node_name}.csv")
        if hasattr(node, '_communication_protocol') and node._communication_protocol:
            if hasattr(node._communication_protocol, 'comm_logger') and node._communication_protocol.comm_logger:
                node._communication_protocol.comm_logger.set_file_path(comm_log_path, auto_save=True)
        
        nodes.append(node)
    
    print(f"📊 Communication logs will be saved to: {comm_log_dir}")

    try:
        # Connect nodes
        topology = network_config.get("topology")
        if not topology:
            raise ValueError("Missing 'topology' configuration in YAML file.")
        if n > Settings.gossip.TTL:
            print(
                f""""TTL less than the number of nodes ({Settings.gossip.TTL} < {n}).
                Some messages will not be delivered depending on the topology."""
            )
        
        # If using D-Cliques topology, build label-aware cliques and custom adjacency matrix
        from p2pfl.learning.aggregators.d_sgd_clique_avg import DSGDCliqueAvg
        if topology in ["dclique_3", "dclique_4", "dclique_5"]:
            # Determine clique size
            clique_size = int(topology.split("_")[1])
            
            # Extract label distributions from partitions
            from collections import Counter
            node_labels = {}
            for i, partition in enumerate(partitions):
                # Get label distribution from partition
                try:
                    # Access the training data
                    if hasattr(partition, '_data'):
                        data = partition._data
                        if hasattr(partition, '_train_split_name'):
                            train_data = data[partition._train_split_name]
                        else:
                            train_data = data
                        
                        # Access raw data without transforms using the underlying data
                        # HuggingFace datasets store raw data in _data attribute
                        if hasattr(train_data, '_data'):
                            # Access the PyArrow table directly
                            raw_labels = train_data._data.column('label').to_pylist()
                        else:
                            # Fallback: temporarily disable transforms
                            old_transform = train_data._format_kwargs.get('transform') if hasattr(train_data, '_format_kwargs') else None
                            train_data.set_transform(None)
                            raw_labels = train_data['label']
                            if old_transform:
                                train_data.set_transform(old_transform)
                        
                        label_counts = Counter(raw_labels)
                        node_labels[f"node_{i}"] = {str(k): float(v) for k, v in label_counts.items()}
                except Exception as e:
                    import traceback
                    print(f"⚠️ Could not extract labels for node {i}: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    # Fallback to sequential assignment
                    node_labels = None
                    break
            
            # Build label-aware cliques if we have label information
            if node_labels:
                print(f"🔬 Building label-aware D-Cliques (clique_size={clique_size})...")
                
                # Build adjacency matrix with label-aware cliques
                node_order = [f"node_{i}" for i in range(n)]
                from p2pfl.utils.d_cliques_p2pfl import build_dcliques_adjacency_matrix
                
                adjacency_matrix, cliques = build_dcliques_adjacency_matrix(
                    node_labels=node_labels,
                    node_order=node_order,
                    clique_size=clique_size,
                    iterations=20000,
                    seed=Settings.general.SEED,
                    inter_mode="small_world",
                    small_world_c=2
                )
                
                # Create clique membership map
                clique_map = {}
                for clique_idx, clique_members in enumerate(cliques):
                    for node_id in clique_members:
                        clique_map[node_id] = clique_members
                
                print(f"✅ Built {len(cliques)} label-aware cliques")
                
                # Set clique members for DSGDCliqueAvg aggregators
                for i, node in enumerate(nodes):
                    if isinstance(node.aggregator, DSGDCliqueAvg):
                        node_id = f"node_{i}"
                        clique_members = clique_map[node_id]
                        # Convert node IDs to addresses
                        clique_addrs = {nodes[node_order.index(nid)].addr for nid in clique_members}
                        node.aggregator.set_clique_members(clique_addrs)
                        print(f"🔗 Node {i} ({node.addr}) in clique with {len(clique_members)} members")
            else:
                # Fallback to sequential assignment
                print(f"⚠️ Using sequential clique assignment (no label information)")
                num_cliques = (n + clique_size - 1) // clique_size
                for i, node in enumerate(nodes):
                    if isinstance(node.aggregator, DSGDCliqueAvg):
                        clique_idx = i // clique_size
                        start = clique_idx * clique_size
                        end = min(start + clique_size, n)
                        clique_addrs = {nodes[j].addr for j in range(start, end)}
                        node.aggregator.set_clique_members(clique_addrs)
        else:
            # For non-dclique topologies, use TopologyFactory
            adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        
        print(f"🔗 Connecting nodes with topology...")
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)
        print(f"✅ Nodes connected, waiting for convergence...")
        try:
            wait_convergence(nodes, n - 1, only_direct=False, wait=60, debug=False)  # type: ignore
            print(f"✅ Convergence complete")
        except Exception as e:
            print(f"❌ Error during convergence: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Additional connections
        additional_connections = network_config.get("additional_connections")
        if additional_connections:
            for source, connect_to in additional_connections:
                nodes[source].connect(nodes[connect_to].addr)

        # Start Learning
        r = experiment_config.get("rounds")
        e = experiment_config.get("epochs")
        trainset_size = experiment_config.get("trainset_size")
        print(f"🚀 Starting learning: {r} rounds, {e} epochs, trainset_size={trainset_size}")
        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        # Start Learning
        print(f"📡 Sending start learning command to node 0...")
        nodes[0].set_start_learning(rounds=r, epochs=e, trainset_size=trainset_size)
        print(f"✅ Start learning command sent")

        # Wait and check
        # Get wait_timeout from experiment config (in minutes), default to 60 minutes (1 hour)
        wait_timeout = experiment_config.get("wait_timeout", 60)
        wait_to_finish(nodes, timeout=wait_timeout * 60, debug=debug)  # Convert minutes to seconds

    except Exception as e:
        raise e
    finally:
        # Save communication logs before stopping nodes
        print(f"\n💾 Saving communication logs to: {comm_log_dir}")
        for i, node in enumerate(nodes):
            if hasattr(node, '_communication_protocol') and node._communication_protocol:
                if hasattr(node._communication_protocol, 'comm_logger') and node._communication_protocol.comm_logger:
                    try:
                        node._communication_protocol.comm_logger.save()
                        print(f"  ✓ Saved logs for node_{i}")
                    except Exception as save_err:
                        print(f"  ✗ Failed to save logs for node_{i}: {save_err}")
        
        # Stop Nodes
        for node in nodes:
            node.stop()
        # Profiling
        if start_time:
            print(f"Execution time: {time.time() - start_time} seconds")
        if profiling_enabled:
            # Stop profiler
            yappi.stop()
            # Save stats
            profile_dir = os.path.join(profiling_output_dir, str(uuid.uuid4()))
            os.makedirs(profile_dir, exist_ok=True)
            for thread in yappi.get_thread_stats():
                yappi.get_func_stats(ctx_id=thread.id).save(f"{profile_dir}/{thread.name}-{thread.id}.pstat", type="pstat")
            # Print where the stats were saved
            print(f"Profile stats saved in {profile_dir}")
