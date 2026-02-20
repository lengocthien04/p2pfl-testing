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

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
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
        """
        Wait for all nodes to reach the current round before advancing.
        
        This prevents fast nodes from moving to round N+1 while slow nodes are still on round N,
        which causes "late round" errors and system deadlock.
        """
        if state.round is None:
            return
        
        current_round = state.round
        wait_time = Settings.heartbeat.WAIT_CONVERGENCE
        
        logger.info(state.addr, f"⏸️  Waiting for all nodes to finish round {current_round} (max {wait_time}s)...")
        
        start_time = time.time()
        check_interval = 1.0  # Check every second
        
        while time.time() - start_time < wait_time:
            # Check if all neighbors have reached the current round
            all_synced = True
            neighbors = communication_protocol.get_neighbors(only_direct=False)
            
            for nei_addr in neighbors:
                nei_round = state.nei_status.get(nei_addr, -1)
                if nei_round < current_round:
                    all_synced = False
                    break
            
            if all_synced:
                elapsed = time.time() - start_time
                logger.info(state.addr, f"✅ All nodes synced at round {current_round} (waited {elapsed:.1f}s)")
                return
            
            # Wait before checking again
            time.sleep(check_interval)
        
        # Timeout - log warning but continue
        elapsed = time.time() - start_time
        unsynced = [addr for addr, r in state.nei_status.items() if r < current_round]
        logger.warning(
            state.addr,
            f"⚠️  Round sync timeout after {elapsed:.1f}s. Unsynced nodes: {unsynced}. Continuing anyway..."
        )

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
