#
# This file is part of the federated_learning_p2p (p2pfl) distribution
#

"""Decentralized SGD (D-SGD) Aggregator (mixing step)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class DSGD(Aggregator):
    """
    D-SGD mixing aggregator.

    It computes:
        theta <- sum_k w_k * theta_k

    - By default: uniform weights (1/K), where K = number of received models.
    - Optional: provide explicit weights via constructor or set_weights().

    IMPORTANT:
    - This only defines the *mixing/aggregation* rule.
    - Whether you are mixing "neighbors only" or "whole network"
      depends on the workflow deciding which models are collected into `models`.
    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, disable_partial_aggregation: bool = False, weights: Optional[List[float]] = None) -> None:
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self._weights: Optional[List[float]] = weights

    def set_weights(self, weights: Optional[List[float]]) -> None:
        """Override weights used in the next aggregation."""
        self._weights = weights

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models (mixing).

        Args:
            models: list of P2PFLModel to mix.

        Returns:
            A P2PFLModel with mixed parameters.

        Raises:
            NoModelsToAggregateError: if no models exist.
            ValueError: if weights invalid.
        """
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")

        k = len(models)

        # Weights
        if self._weights is None:
            w = np.full((k,), 1.0 / float(k), dtype=np.float64)
        else:
            if len(self._weights) != k:
                raise ValueError(f"DSGD weights length {len(self._weights)} must match number of models {k}")
            w = np.asarray(self._weights, dtype=np.float64)
            s = float(np.sum(w))
            if s <= 0:
                raise ValueError("DSGD weights must sum to a positive value")
            w = w / s  # normalize defensively

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

        # Keep sample metadata consistent with FedAvg style (not used for mixing here)
        total_samples = sum([m.get_num_samples() for m in models])

        return models[0].build_copy(params=accum, num_samples=total_samples, contributors=contributors)
