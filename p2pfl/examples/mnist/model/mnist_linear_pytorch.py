#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
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

"""MnistLinear - single linear layer for MNIST (logistic regression)."""

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Metric

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed


class MnistLinear(L.LightningModule):
    """Single linear layer for MNIST classification."""

    def __init__(
        self,
        num_classes: int = 10,
        metric: type[Metric] = Accuracy,
        lr_rate: float = 0.001,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.lr_rate = lr_rate
        if num_classes == 1:
            self.metric = metric(task="binary")
        else:
            self.metric = metric(task="multiclass", num_classes=num_classes)
        self.fc = torch.nn.Linear(28 * 28, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.fc(x), dim=1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        return torch.optim.SGD(self.parameters(), lr=self.lr_rate, momentum=0.9)

    def training_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Training step."""
        x = batch["image"].float()
        y = batch["label"]
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Validate step not implemented."""
        raise NotImplementedError("Validation step not implemented")

    def test_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Test step."""
        x = batch["image"].float()
        y = batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)
        return loss


def model_build_fn(*args, **kwargs) -> LightningModel:
    """Export the model build function."""
    compression = kwargs.pop("compression", None)
    return LightningModel(MnistLinear(*args, **kwargs), compression=compression)
