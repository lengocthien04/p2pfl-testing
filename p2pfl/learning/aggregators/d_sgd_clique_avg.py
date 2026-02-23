#
# This file is part of the federated_learning_p2p (p2pfl) distribution
#

"""D-SGD with Clique Averaging - two-stage weighted averaging approximation."""

from __future__ import annotations

from typing import Set

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class DSGDCliqueAvg(Aggregator):
    """
    D-SGD with Clique Averaging (practical approximation).
    
    This is a practical approximation of the D-Cliques paper's Algorithm 3.
    Instead of exchanging gradients separately, we use a two-stage weighted averaging:
    
    1. First stage: Average models within clique (uniform weights)
    2. Second stage: Average clique result with non-clique neighbors (uniform weights)
    
    The paper's full algorithm requires:
    1. Exchange gradients within clique
    2. Compute clique-averaged gradient
    3. Take gradient step
    4. Average models with all neighbors
    
    This approximation:
    1. Take gradient step (standard training)
    2. Average models within clique first
    3. Then average clique result with non-clique neighbors
    
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
            clique_weight_ratio: Not used in two-stage approach, kept for compatibility
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
        Aggregate models with two-stage clique-aware averaging.
        
        Algorithm:
        1. Separate models into clique and non-clique
        2. Average clique models uniformly (stage 1)
        3. Average clique result with non-clique models uniformly (stage 2)
        
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

        # Separate clique and non-clique models
        clique_models = []
        non_clique_models = []
        
        for model in models:
            contributors = model.get_contributors()
            is_clique_member = any(c in self._clique_members for c in contributors)
            
            if is_clique_member:
                clique_models.append(model)
            else:
                non_clique_models.append(model)

        from p2pfl.management.logger import logger
        logger.info(self.addr, f"🔢 Clique averaging: {len(clique_models)} clique, {len(non_clique_models)} non-clique")

        # Stage 1: Average clique models
        if len(clique_models) > 0:
            clique_averaged = self._uniform_average(clique_models)
        else:
            # No clique models, just use first model as placeholder
            clique_averaged = None

        # Stage 2: Average with non-clique models
        if clique_averaged is not None and len(non_clique_models) > 0:
            # Both clique and non-clique: average them together
            all_models = [clique_averaged] + non_clique_models
            return self._uniform_average(all_models)
        elif clique_averaged is not None:
            # Only clique models
            return clique_averaged
        elif len(non_clique_models) > 0:
            # Only non-clique models
            return self._uniform_average(non_clique_models)
        else:
            # Should never happen (caught by len(models) == 0 check)
            raise NoModelsToAggregateError("No models to aggregate")

    def _uniform_average(self, models: list[P2PFLModel]) -> P2PFLModel:
        """Uniform averaging of models (1/K weight for each)."""
        k = len(models)
        first_params = models[0].get_parameters()
        accum = [np.zeros_like(layer) for layer in first_params]
        
        for model in models:
            params = model.get_parameters()
            for i, layer in enumerate(params):
                accum[i] += layer / k
        
        contributors = []
        for m in models:
            contributors += m.get_contributors()
        
        total_samples = sum([m.get_num_samples() for m in models])
        
        return models[0].build_copy(
            params=accum,
            num_samples=total_samples,
            contributors=contributors
        )
