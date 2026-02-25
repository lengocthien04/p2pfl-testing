#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#

"""Lightweight model container for received parameter snapshots."""

from __future__ import annotations

from typing import Any

import numpy as np

from p2pfl.learning.compression.manager import CompressionManager


class LightweightModelSnapshot:
    """
    Lightweight model-like container used for aggregation of received models.

    It stores decoded parameters and metadata only, avoiding full framework model
    deep copies when handling incoming network updates.
    """

    def __init__(
        self,
        params: list[np.ndarray],
        num_samples: int,
        contributors: list[str],
        additional_info: dict[str, Any] | None = None,
        encoded_params: bytes | None = None,
        compression: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._params = params
        self._num_samples = num_samples
        self._contributors = contributors
        self._additional_info = additional_info or {}
        self._encoded_params = encoded_params
        self._compression = compression or {}

    def get_parameters(self) -> list[np.ndarray]:
        """Return stored parameters."""
        return self._params

    def get_contributors(self) -> list[str]:
        """Return contributors."""
        if self._contributors == []:
            raise ValueError("Contributors are empty")
        return self._contributors

    def get_num_samples(self) -> int:
        """Return number of samples."""
        if self._num_samples == 0:
            raise ValueError("Number of samples required")
        return self._num_samples

    def get_info(self, callback: str | None = None) -> Any:
        """Return additional model info."""
        if callback is None:
            return self._additional_info
        return self._additional_info[callback]

    def add_info(self, callback: str, info: Any) -> None:
        """Add callback info."""
        self._additional_info[callback] = info

    def encode_parameters(self) -> bytes:
        """
        Encode stored parameters.

        If original encoded payload is available, reuse it to avoid recompression and
        preserve the exact wire format of forwarded updates.
        """
        if self._encoded_params is not None:
            return self._encoded_params
        return CompressionManager.apply(self._params, self._additional_info, self._compression)
