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
"""Train stage."""

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


class TrainStage(Stage):
    """Train stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "TrainStage"

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
            raise Exception("Invalid parameters on TrainStage.")

        try:
            check_early_stop(state)

            # Set Models To Aggregate
            from p2pfl.settings import Settings
            if Settings.training.NEIGHBOR_ONLY_AGGREGATION:
                # Accept models from all trainset (for gossip to work)
                aggregator.set_nodes_to_aggregate(state.train_set)
                # But only aggregate from neighbors
                direct_neighbors_dict = communication_protocol.get_neighbors(only_direct=True)
                direct_neighbors = list(direct_neighbors_dict.keys())
                neighbors_in_trainset = [n for n in direct_neighbors if n in state.train_set]
                nodes_to_actually_aggregate = set([state.addr] + neighbors_in_trainset)
                # Set filter on aggregator
                aggregator.neighbor_filter = nodes_to_actually_aggregate
                logger.info(state.addr, f"🎯 Neighbor-only: accepting all, aggregating from {len(nodes_to_actually_aggregate)} neighbors")
            else:
                # Aggregate from all trainset (fully connected)
                aggregator.set_nodes_to_aggregate(state.train_set)
                aggregator.neighbor_filter = None

            check_early_stop(state)

            # Evaluate and send metrics
            TrainStage.__evaluate(state, learner, communication_protocol)

            check_early_stop(state)

            # Train
            logger.info(state.addr, "🏋️‍♀️ Training...")
            learner.fit()
            logger.info(state.addr, "🎓 Training done.")

            check_early_stop(state)

            # Aggregate Model
            models_added = aggregator.add_model(learner.get_model())

            # send model added msg to ALL nodes (not just direct neighbors)
            agg_msg = communication_protocol.build_msg(
                ModelsAggregatedCommand.get_name(),
                models_added,
                round=state.round,
            )
            all_neis = communication_protocol.get_neighbors(only_direct=False)
            for addr, (client, _) in all_neis.items():
                if addr != state.addr:
                    client.send(agg_msg, temporal_connection=True)
            TrainStage.__gossip_model_aggregation(state, communication_protocol, aggregator)

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
        Gossip model aggregation.

        CAREFULL:
            - Full connected trainset to increase aggregation speed. On real scenarios, this won't
            be possible, private networks and firewalls.
            - Needed because the trainset can split the networks (and neighbors that are not in the
            trainset won't receive the aggregation).
        """

        # Cache for encoded models: key = frozenset of contributors, value = (model_msg, contributors)
        _model_cache: dict[frozenset, tuple[Any, list[str]]] = {}

        # Anonymous functions
        def early_stopping_fn():
            return state.round is None

        def get_candidates_fn() -> list[str]:
            candidates = set(state.train_set) - {state.addr}
            return [n for n in candidates if len(TrainStage.__get_remaining_nodes(n, state)) != 0]

        def status_fn() -> Any:
            return [
                (
                    n,
                    TrainStage.__get_aggregated_models(n, state),
                )  # reemplazar por Aggregator - borrarlo de node
                for n in communication_protocol.get_neighbors(only_direct=False)
                if (n in state.train_set)
            ]

        def model_fn(node: str) -> tuple[Any, str, int, list[str]]:
            if state.round is None:
                raise Exception("Round not initialized.")
            except_nodes = TrainStage.__get_aggregated_models(node, state)
            # Cache key: what the target already has determines the model we send
            cache_key = frozenset(except_nodes)
            if cache_key in _model_cache:
                cached = _model_cache[cache_key]
                if cached is None:
                    return (None, PartialModelCommand.get_name(), state.round, [])
                cached_msg, cached_contributors = cached
                return (cached_msg, PartialModelCommand.get_name(), state.round, cached_contributors)
            try:
                model = aggregator.get_model(except_nodes)
            except NoModelsToAggregateError:
                logger.debug(state.addr, f"❔ No models to aggregate for {node}.")
                _model_cache[cache_key] = None
                return (None, PartialModelCommand.get_name(), state.round, [])
            contributors = model.get_contributors()
            # Expensive: encode_parameters (pickle ~4MB) + build_weights (protobuf wrap)
            model_msg = communication_protocol.build_weights(
                PartialModelCommand.get_name(),
                state.round,
                model.encode_parameters(),
                contributors,
                model.get_num_samples(),
            )
            _model_cache[cache_key] = (model_msg, contributors)
            return (model_msg, PartialModelCommand.get_name(), state.round, contributors)

        # Gossip
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
        return set(state.train_set) - set(TrainStage.__get_aggregated_models(node, state))
