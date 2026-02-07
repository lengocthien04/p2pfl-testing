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
import time

import torch
from p2pfl.communication.commands.message.metrics_command import MetricsCommand
# from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory
from p2pfl.communication.commands.weights.full_model_command import FullModelCommand


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

            # Promote future models received during previous round transitions
            with state.incoming_models_lock:
                if state.round in state.future_incoming_models:
                    logger.info(state.addr, f"⏩ Promoting {len(state.future_incoming_models[state.round])} future models.")
                    state.incoming_models_buffer.extend(state.future_incoming_models[state.round])
                    del state.future_incoming_models[state.round]

            # Set Models To Aggregate
            aggregator.set_nodes_to_aggregate(state.train_set)

            check_early_stop(state)

            # Evaluate and send metrics
            TrainStage.__evaluate(state, learner, communication_protocol)

            check_early_stop(state)

            # Train
            logger.info(state.addr, "🏋️‍♀️ Training...")
            # Block any network thread from calling learner.set_model during backward()
            with state.model_update_lock:
                learner.fit()
            logger.info(state.addr, "🎓 Training done.")

            check_early_stop(state)
            
            # 1. Capture local model for sending
            encoded = learner.get_model().encode_parameters()
            
            # 2. Add self to aggregator
            # FIX: Must copy the model, otherwise buffer processing will overwrite 'Self' weights
            self_model = learner.get_model()
            self_copy = self_model.build_copy(params=self_model.get_parameters(), 
                                              num_samples=self_model.get_num_samples(), 
                                              contributors=self_model.get_contributors())
            aggregator.add_model(self_copy)
            
            # 3. Process any models buffered during training
            # Loop to ensure we catch models that arrive WHILE we are processing the buffer
            while True:
                with state.incoming_models_lock:
                    if not state.incoming_models_buffer:
                        break
                    buffered_models = state.incoming_models_buffer[:]
                    state.incoming_models_buffer.clear()
                
                # We MUST hold the lock here. If we don't, FullModelCommand might try 
                # to update the learner at the same time, or buffer more items that we miss.
                with state.model_update_lock:
                    logger.info(state.addr, f"🔄 Processing {len(buffered_models)} buffered models.")
                    for item in buffered_models:
                        # Use learner to decode
                        learner.set_model(item["weights"])
                        model = learner.get_model()
                        
                        # Apply metadata logic
                        src = item["source"]
                        if hasattr(model, "set_contributors"): model.set_contributors([src])
                        elif hasattr(model, "contributors"): model.contributors = [src]
                        elif hasattr(model, "_contributors"): model._contributors = [src]

                        ns = item["kwargs"].get("num_samples", None)
                        if ns is not None:
                            if hasattr(model, "set_num_samples"): model.set_num_samples(ns)
                            elif hasattr(model, "num_samples"): model.num_samples = ns
                            elif hasattr(model, "_num_samples"): model._num_samples = ns
                        
                        # FIX: Copy model to prevent reference issues
                        model_copy = model.build_copy(params=model.get_parameters(),
                                                      num_samples=model.get_num_samples(),
                                                      contributors=model.get_contributors())
                        aggregator.add_model(model_copy)

            # Send my FULL model to direct neighbors (sync)
            msg = communication_protocol.build_weights(
                FullModelCommand.get_name(),
                state.round,
                encoded,
            )

            # send to direct neighbors that are in train_set
            for n in state.train_set:
                if n != state.addr:
                    # Retry loop for robustness
                    for attempt in range(3):
                        try:
                            communication_protocol.send(n, msg)
                            break # Success
                        except Exception as e:
                            if attempt < 2:
                                logger.warning(state.addr, f"⚠️ Send to {n} failed (attempt {attempt+1}). Retrying... ({e})")
                                try:
                                    communication_protocol.connect(n)
                                    time.sleep(0.5)
                                except: pass
                            else:
                                logger.error(state.addr, f"❌ Failed to send model to {n} after retries: {e}")
            # TODO: print("Broadcast redundante")
            # communication_protocol.broadcast(
            #     communication_protocol.build_msg(
            #         ModelsAggregatedCommand.get_name(),
            #         models_added,
            #         round=state.round,
            #     )
            # )
            # TrainStage.__gossip_model_aggregation(state, communication_protocol, aggregator)

            check_early_stop(state)

            # Set aggregated model
            # Debug log to see who we are waiting for
            logger.info(state.addr, f"⏳ Waiting for aggregation. Expecting: {len(state.train_set)} nodes.")
            with torch.no_grad():
                agg_model = aggregator.wait_and_get_aggregation()
            
            with state.model_update_lock:
                learner.set_model(agg_model)

            # Save communication logs after every round
            if hasattr(communication_protocol, "save_logs"):
                communication_protocol.save_logs()

            # Share that aggregation is done
            # communication_protocol.broadcast(communication_protocol.build_msg(ModelsReadyCommand.get_name(), [], round=state.round))

            # Next stage
            return StageFactory.get_stage("RoundFinishedStage")
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
                for n in communication_protocol.get_neighbors(only_direct=True)
                if (n in state.train_set)
            ]

        def model_fn(node: str) -> tuple[Any, str, int, list[str]]:
            if state.round is None:
                raise Exception("Round not initialized.")
            try:
                model = aggregator.get_model(TrainStage.__get_aggregated_models(node, state))
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

        # Gossip
        communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            create_connection=False,
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
