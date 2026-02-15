#
# This file is part of the federated_learning_p2p (p2pfl) distribution
#

"""D-SGD with Clique Averaging - approximation using weighted mixing."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class DSGDCliqueAvg(Aggregator):
    """
    D-SGD with Clique Averaging (approximation).
    
    This is a practical approximation of the D-Cliques paper's Algorithm 3.
    Instead of exchanging gradients separately, we use weighted model averaging
    where clique members get higher weights than inter-clique neighbors.
    
    The paper's full algorithm requires:
    1. Exchange gradients within clique
    2. Compute clique-averaged gradient
    3. Take gradient step
    4. Average models with all neighbors
    
    This approximation:
    1. Take gradient step (standard training)
    2. Average models with weighted mixing (higher weight for clique members)
    
    Clique membership must be set via set_clique_members() before aggregation.
    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(
        self, 
        disable_partial_aggregation: bool = False,
        clique_weight_ratio: float = 2.0
    ) -> None:
        """
        Initialize D-SGD with Clique Averaging.
        
        Args:
            disable_partial_aggregation: Whether to disable partial aggregation
            clique_weight_ratio: Ratio of weight for clique members vs inter-clique neighbors
                                Higher values give more weight to clique members (default: 2.0)
        """
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self._clique_members: Set[str] = set()
        self._clique_weight_ratio = clique_weight_ratio

    def set_clique_members(self, clique_members: Set[str]) -> None:
        """
        Set the clique members for this node.
        
        Args:
            clique_members: Set of node addresses that are in the same clique
        """
        self._clique_members = clique_members

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate models with clique-aware weighting.
        
        Args:
            models: list of P2PFLModel to mix
            
        Returns:
            A P2PFLModel with mixed parameters
            
        Raises:
            NoModelsToAggregateError: if no models exist
        """
        if len(models) == 0:
            raise NoModelsToAggregateError(
                f"({self.addr}) Trying to aggregate models when there is no models"
            )

        k = len(models)
        
        # Compute weights based on clique membership
        weights = []
        for model in models:
            contributors = model.get_contributors()
            # Check if any contributor is in the clique
            is_clique_member = any(c in self._clique_members for c in contributors)
            
            if is_clique_member:
                weights.append(self._clique_weight_ratio)
            else:
                weights.append(1.0)
        
        # Normalize weights
        w = np.array(weights, dtype=np.float64)
        w = w / np.sum(w)
        
        # Accumulator
        first_params = models[0].get_parameters()
        accum = [np.zeros_like(layer) for layer in first_params]

        # Weighted sum
        for m_idx, m in enumerate(models):
            params = m.get_parameters()
            for i, layer in enumerate(params):
                accum[i] = accum[i] + (layer * w[m_idx])

        # Contributors union
        contributors: list[str] = []
        for m in models:
            contributors += m.get_contributors()

        # Keep sample metadata
        total_samples = sum([m.get_num_samples() for m in models])

        return models[0].build_copy(
            params=accum, 
            num_samples=total_samples, 
            contributors=contributors
        )
