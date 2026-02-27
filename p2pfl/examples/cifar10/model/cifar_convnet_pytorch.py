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

"""CifarConvNet - lightweight CNN for CIFAR-10."""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Metric

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed


class CifarConvNet(L.LightningModule):
    """Lightweight 3-layer CNN for CIFAR-10 classification."""

    def __init__(
        self,
        num_classes: int = 10,
        metric: type[Metric] = Accuracy,
        lr_rate: float = 0.002,
    ) -> None:
        """Initialize the CifarConvNet model."""
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.lr_rate = lr_rate
        if num_classes == 1:
            self.metric = metric(task="binary")
        else:
            self.metric = metric(task="multiclass", num_classes=num_classes)

        # Conv block 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Conv block 2: 32 -> 64, with maxpool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        # Conv block 3: 64 -> 128, with maxpool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        # Classifier: 128*8*8 -> 256 -> 10
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if len(x.shape) == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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
    return LightningModel(CifarConvNet(*args, **kwargs), compression=compression)
