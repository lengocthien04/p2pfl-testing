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

"""Round Finished Stage."""

import time

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class RoundFinishedStage(Stage):
    """Round Finished Stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "RoundFinishedStage"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        learner: Learner | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        aggregator: Aggregator | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if state is None or communication_protocol is None or aggregator is None or learner is None:
            raise Exception("Invalid parameters on RoundFinishedStage.")

        # Set Next Round
        aggregator.clear()

        # Wait for all nodes to finish this round before advancing
        RoundFinishedStage.__wait_round_sync(state, communication_protocol)

        state.increase_round()

        # Next Step or Finish
        logger.info(
            state.addr,
            f"🎉 Round {state.round} of {state.total_rounds} finished.",
        )
        if state.round is None or state.total_rounds is None:
            raise ValueError("Round or total rounds not set.")

        if state.round < state.total_rounds:
            return StageFactory.get_stage("VoteTrainSetStage")
        else:
            # At end, all nodes compute metrics
            RoundFinishedStage.__evaluate(state, learner, communication_protocol)
            # Finish
            state.clear()
            logger.info(state.addr, "😋 Training finished!!")
            return None

    @staticmethod
    def __wait_round_sync(state: NodeState, communication_protocol: CommunicationProtocol) -> None:
        """Wait for all trainset nodes to reach the current round before advancing."""
        if state.round is None or not state.train_set:
            return

        current_round = state.round
        wait_time = Settings.heartbeat.WAIT_CONVERGENCE

        logger.info(state.addr, f"⏸️  Waiting for trainset to finish round {current_round} (max {wait_time}s)...")

        start_time = time.time()

        while time.time() - start_time < wait_time:
            # Check all trainset nodes (except self)
            all_synced = True
            for node in state.train_set:
                if node == state.addr:
                    continue
                nei_round = state.nei_status.get(node, -1)
                # CRITICAL: Check for EXACT round match, not >=
                # This prevents stale nei_status from previous rounds causing premature barrier pass
                if nei_round != current_round:
                    all_synced = False
                    break

            if all_synced:
                elapsed = time.time() - start_time
                logger.info(state.addr, f"✅ All trainset synced at round {current_round} ({elapsed:.1f}s)")
                return

            time.sleep(1.0)

        # Timeout
        unsynced = [n for n in state.train_set if n != state.addr and state.nei_status.get(n, -1) != current_round]
        logger.info(state.addr, f"⚠️ Round sync timeout. Unsynced: {len(unsynced)} nodes: {unsynced}. Continuing...")

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
