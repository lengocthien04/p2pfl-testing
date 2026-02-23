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
"""Train stage with neighbor-only aggregation for true D-SGD."""

from typing import Any

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory


class TrainStageNeighborOnly(Stage):
    """Train stage with neighbor-only aggregation for true D-SGD on sparse topologies."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "TrainStageNeighborOnly"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        learner: Learner | None = None,
        aggregator: Aggregator | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if state is None or communication_protocol is None or aggregator is None or learner is None:
            raise Exception("Invalid parameters on TrainStageNeighborOnly.")

        try:
            check_early_stop(state)

            # CRITICAL CHANGE: Aggregate only from direct neighbors + self WHO ARE IN TRAINSET
            # This is true D-SGD behavior for sparse topologies
            direct_neighbors_dict = communication_protocol.get_neighbors(only_direct=True)
            direct_neighbors = list(direct_neighbors_dict.keys())  # Convert dict to list of addresses
            
            # CRITICAL FIX: Only aggregate from neighbors that are in trainset
            neighbors_in_trainset = [n for n in direct_neighbors if n in state.train_set]
            nodes_to_aggregate = [state.addr] + neighbors_in_trainset
            
            logger.info(state.addr, f"🎯 Aggregating from {len(nodes_to_aggregate)} neighbors in trainset (including self)")
            logger.info(state.addr, f"   Direct neighbors: {len(direct_neighbors)}, In trainset: {len(neighbors_in_trainset)}")
            aggregator.set_nodes_to_aggregate(nodes_to_aggregate)

            check_early_stop(state)

            # Evaluate and send metrics
            TrainStageNeighborOnly.__evaluate(state, learner, communication_protocol)

            check_early_stop(state)

            # Train
            logger.info(state.addr, "🏋️‍♀️ Training...")
            learner.fit()
            logger.info(state.addr, "🎓 Training done.")

            check_early_stop(state)

            # Aggregate Model
            models_added = aggregator.add_model(learner.get_model())

            # Broadcast model added
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    ModelsAggregatedCommand.get_name(),
                    models_added,
                    round=state.round,
                )
            )
            
            # Gossip models - use same logic as TrainStage (help everyone, gossip to everyone)
            TrainStageNeighborOnly.__gossip_model_aggregation(state, communication_protocol, aggregator)

            check_early_stop(state)

            # Set aggregated model
            agg_model = aggregator.wait_and_get_aggregation()
            learner.set_model(agg_model)

            # Share that aggregation is done
            communication_protocol.broadcast(communication_protocol.build_msg(ModelsReadyCommand.get_name(), [], round=state.round))

            # Next stage
            return StageFactory.get_stage("GossipModelStage")
        except EarlyStopException:
            return None

    @staticmethod
    def __evaluate(state: NodeState, learner: Learner, communication_protocol: CommunicationProtocol) -> None:
        logger.info(state.addr, "🔬 Evaluating...")
        results = learner.evaluate()
        logger.info(state.addr, f"📈 Evaluated. Results: {results}")
        # Send metrics
        if len(results) > 0:
            logger.info(state.addr, "📢 Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=state.round,
                )
            )

    @staticmethod
    def __gossip_model_aggregation(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        """
        Gossip model aggregation - SAME AS TrainStage.
        
        Help all nodes in trainset get their models (not just neighbors).
        This prevents deadlock while still only aggregating from neighbors.
        """

        # Anonymous functions - COPIED FROM TrainStage
        def early_stopping_fn():
            return state.round is None

        def get_candidates_fn() -> list[str]:
            candidates = set(state.train_set) - {state.addr}
            return [n for n in candidates if len(TrainStageNeighborOnly.__get_remaining_nodes(n, state)) != 0]

        def status_fn() -> Any:
            return [
                (
                    n,
                    TrainStageNeighborOnly.__get_aggregated_models(n, state),
                )
                for n in communication_protocol.get_neighbors(only_direct=False)
                if (n in state.train_set)
            ]

        def model_fn(node: str) -> tuple[Any, str, int, list[str]]:
            if state.round is None:
                raise Exception("Round not initialized.")
            try:
                model = aggregator.get_model(TrainStageNeighborOnly.__get_aggregated_models(node, state))
            except NoModelsToAggregateError:
                logger.debug(state.addr, f"❔ No models to aggregate for {node}.")
                return (
                    None,
                    PartialModelCommand.get_name(),
                    state.round,
                    [],
                )
            model_msg = communication_protocol.build_weights(
                PartialModelCommand.get_name(),
                state.round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )
            return (
                model_msg,
                PartialModelCommand.get_name(),
                state.round,
                model.get_contributors(),
            )

        # Gossip - SAME AS TrainStage
        communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            create_connection=True,
        )

    @staticmethod
    def __get_aggregated_models(node: str, state: NodeState) -> list[str]:
        try:
            return state.models_aggregated[node]
        except KeyError:
            return []

    @staticmethod
    def __get_remaining_nodes(node: str, state: NodeState) -> set[str]:
        """
        Get remaining nodes that this node needs - NEIGHBOR-AWARE VERSION.
        
        Unlike TrainStage, nodes only need models from their direct neighbors (not all trainset).
        We need to check what models 'node' actually needs based on its neighbors.
        
        Problem: We don't have access to communication_protocol here to get node's neighbors.
        Solution: Assume if a node has collected enough models, it's done.
        """
        # Get what node has collected
        collected = set(TrainStageNeighborOnly.__get_aggregated_models(node, state))
        
        # In neighbor-only mode, we can't know exactly which neighbors 'node' needs
        # But we know: if node has collected models and stopped asking, it's done
        # For now, return empty if node has ANY models (it will handle its own aggregation)
        # This is a heuristic - gossip will exit when no node is asking for help
        
        # Better heuristic: return trainset - collected (same as TrainStage)
        # The node itself will only aggregate from its neighbors via set_nodes_to_aggregate
        return set(state.train_set) - collected
