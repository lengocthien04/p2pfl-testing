#
# This file is part of the federated_learning_p2p (p2pfl) distribution
#

"""D-SGD with Clique Averaging - two-stage weighted averaging approximation."""

from __future__ import annotations

from typing import Set

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger


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
        1. Filter models by neighbor_filter (if set) - only keep neighbor models
        2. Separate clique models from all filtered models
        3. Average clique models uniformly (stage 1) → produces clique-averaged model
        4. Average clique-averaged model with ALL filtered neighbor models (stage 2)
        
        Example: Node in clique with 3 members, has 4 direct neighbors (3 clique + 1 outside)
        - Filter: Keep only models from 4 direct neighbors
        - Stage 1: Average 3 clique models → 1 clique-averaged model
        - Stage 2: Average 1 clique result + 4 neighbor models = 5 total models
        
        Args:
            models: list of P2PFLModel to mix
            
        Returns:
            A P2PFLModel with mixed parameters
            
        Raises:
            NoModelsToAggregateError: if no models exist
        """
        models = self._filter_by_neighbors(models)

        if len(models) == 0:
            raise NoModelsToAggregateError(
                f"({self.addr}) Trying to aggregate models when there is no models"
            )

        # Keep ALL filtered models as neighbors for stage 2
        all_neighbor_models = models.copy()
        
        # Separate clique models for stage 1
        clique_models = []
        
        for model in models:
            contributors = model.get_contributors()
            is_clique_member = any(c in self._clique_members for c in contributors)
            
            if is_clique_member:
                clique_models.append(model)

        logger.info(self.addr, f"🔢 Clique averaging: {len(clique_models)} clique models, {len(all_neighbor_models)} total neighbor models")

        # Stage 1: Average clique models to get single clique-averaged model
        if len(clique_models) > 0:
            clique_averaged = self._uniform_average(clique_models)
            logger.info(self.addr, f"✅ Stage 1: Averaged {len(clique_models)} clique models")
        else:
            # No clique models - just do regular D-SGD averaging
            logger.info(self.addr, f"⚠️ No clique models, using regular D-SGD")
            return self._uniform_average(all_neighbor_models)

        # Stage 2: Average clique-averaged model with ALL neighbor models (D-SGD style)
        # This includes the original clique models that are neighbors
        all_models_for_stage2 = [clique_averaged] + all_neighbor_models
        logger.info(self.addr, f"✅ Stage 2: Averaging 1 clique result + {len(all_neighbor_models)} neighbors = {len(all_models_for_stage2)} total")
        return self._uniform_average(all_models_for_stage2)

    def _uniform_average(self, models: list[P2PFLModel]) -> P2PFLModel:
        """Uniform averaging of models (1/K weight for each)."""
        k = len(models)
        template_model = self._get_template_model(models)
        first_params = models[0].get_parameters()
        # Use float64 arrays to avoid casting issues
        accum = [np.zeros_like(layer, dtype=np.float64) for layer in first_params]
        
        for model in models:
            params = model.get_parameters()
            for i, layer in enumerate(params):
                # Convert to float64 for averaging
                accum[i] += layer.astype(np.float64) / k
        
        # Convert back to original dtypes
        result_params = []
        for i, layer in enumerate(first_params):
            result_params.append(accum[i].astype(layer.dtype))
        
        contributors = []
        for m in models:
            contributors += m.get_contributors()
        
        total_samples = sum([m.get_num_samples() for m in models])
        
        return template_model.build_copy(
            params=result_params,
            num_samples=total_samples,
            contributors=contributors
        )
