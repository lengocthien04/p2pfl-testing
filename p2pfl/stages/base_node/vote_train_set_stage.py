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
"""Vote Train Set Stage."""

import random

from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import EarlyStopException, Stage
from p2pfl.stages.stage_factory import StageFactory


class VoteTrainSetStage(Stage):
    """Vote Train Set Stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "VoteTrainSetStage"

    @staticmethod
    def execute(
        trainset_size: int | None = None,
        state: NodeState | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        generator: random.Random | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if state is None or communication_protocol is None or trainset_size is None or generator is None:
            raise Exception("Invalid parameters on VoteTrainSetStage.")

        try:
            # Deterministic train set selection — all nodes compute the same result
            # from the shared seed and current round, eliminating vote exchange.
            candidates = list(communication_protocol.get_neighbors(only_direct=False))
            if state.addr not in candidates:
                candidates.append(state.addr)
            candidates.sort()

            samples = min(trainset_size, len(candidates))
            round_generator = random.Random(hash((generator.getstate()[1][0], state.round)))
            state.train_set = round_generator.sample(candidates, samples)

            logger.info(
                state.addr,
                f"🚂 Train set of {len(state.train_set)} nodes: {state.train_set}",
            )

            # Next stage
            if state.addr in state.train_set:
                return StageFactory.get_stage("TrainStage")
            else:
                logger.debug(state.addr, "Node not in train set. Proceeding to WaitAggregatedModelsStage.")
                return StageFactory.get_stage("WaitAggregatedModelsStage")
        except EarlyStopException:
            return None
