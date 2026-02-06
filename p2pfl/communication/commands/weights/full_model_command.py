#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""FullModelCommand."""

from collections.abc import Callable

from p2pfl.communication.commands.command import Command
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class FullModelCommand(Command):
    """FullModelCommand."""

    def __init__(self, state: NodeState, stop: Callable[[], None], aggregator: Aggregator, learner: Learner) -> None:
        """Initialize FullModelCommand."""
        self.state = state
        self.stop = stop
        self.aggregator = aggregator
        self.learner = learner

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "add_model"

    def execute(
        self,
        source: str,
        round: int,
        weights: bytes | None = None,
        **kwargs,
    ) -> None:
        """Execute the command."""
        if weights is None:
            raise ValueError("Weights, contributors and weight are required")

        # # Check if Learning is running
        # if self.state.round is not None:
        #     # Check source
        #     if round != self.state.round:
        #         logger.debug(
        #             self.state.addr,
        #             f"Model reception in a late round ({round} != {self.state.round}).",
        #         )
        #         return
        #     if self.state.aggregated_model_event.is_set():
        #         logger.debug(self.state.addr, "😲 Aggregated model not expected.")
        #         return
        #     try:
        #         logger.info(self.state.addr, "📦 Aggregated model received.")
        #         # Decode and set model
        #         self.learner.set_model(weights)
        #         # Release here caused the simulation to crash before
        #         self.state.aggregated_model_event.set()

        # Check if Learning is running
        if self.state.round is not None:
            # Round guard
            if round < self.state.round:
                logger.debug(
                    self.state.addr,
                    f"Model reception in a late round ({round} < {self.state.round}).",
                )
                return
            elif round > self.state.round:
                logger.info(self.state.addr, f"📥 Buffering future model from {source} for round {round}.")
                with self.state.incoming_models_lock:
                    if round not in self.state.future_incoming_models:
                        self.state.future_incoming_models[round] = []
                    self.state.future_incoming_models[round].append({"source": source, "weights": weights, "kwargs": kwargs})
                return

            try:
                # For sync learning: treat this as a neighbor model contribution
                logger.info(self.state.addr, f"📦 Model received from {source}.")

                # Decode weights into a P2PFLModel instance.
                # We keep learner decoding to reuse framework checks.
                # IMPORTANT: do not mutate model during training/backward
                # FIX: If locked (training), buffer the model to avoid DEADLINE_EXCEEDED
                if self.state.model_update_lock.acquire(blocking=False):
                    try:
                        self.learner.set_model(weights)
                        model = self.learner.get_model()
                        
                        # Set metadata
                        if hasattr(model, "set_contributors"):
                            model.set_contributors([source])
                        elif hasattr(model, "contributors"):
                            model.contributors = [source]
                        elif hasattr(model, "_contributors"):
                            model._contributors = [source]

                        num_samples = kwargs.get("num_samples", None)
                        if num_samples is not None:
                            if hasattr(model, "set_num_samples"):
                                model.set_num_samples(num_samples)
                            elif hasattr(model, "num_samples"):
                                model.num_samples = num_samples
                            elif hasattr(model, "_num_samples"):
                                model._num_samples = num_samples

                        # Feed into aggregator
                        # FIX: Copy model to prevent reference issues
                        model_copy = model.build_copy(params=model.get_parameters(),
                                                      num_samples=model.get_num_samples(),
                                                      contributors=model.get_contributors())
                        self.aggregator.add_model(model_copy)
                    finally:
                        self.state.model_update_lock.release()
                else:
                    # Training is busy, buffer the model
                    logger.info(self.state.addr, f"📥 Buffering model from {source} (Training busy).")
                    with self.state.incoming_models_lock:
                        self.state.incoming_models_buffer.append({
                            "source": source,
                            "weights": weights,
                            "kwargs": kwargs
                        })

            # Warning: these stops can cause a denegation of service attack
            except DecodingParamsError:
                logger.error(self.state.addr, "❌ Error decoding parameters.")
                self.stop()

            except ModelNotMatchingError:
                logger.error(self.state.addr, "❌ Models not matching.")
                self.stop()

            except Exception as e:
                logger.error(self.state.addr, f"❌ Unknown error adding full model: {e}")
                self.stop()
        else:
            # Learning not started yet (or finished). Buffer as future model to avoid dropping early arrivals.
            logger.info(self.state.addr, f"📥 Buffering model from {source} for round {round} (Learning not started).")
            with self.state.incoming_models_lock:
                if round not in self.state.future_incoming_models:
                    self.state.future_incoming_models[round] = []
                self.state.future_incoming_models[round].append({"source": source, "weights": weights, "kwargs": kwargs})
