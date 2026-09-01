# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import List

import math
import torch
from torch.optim import Optimizer


class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            max_epochs: int,
            warmup_start_lr: float = 0.0,
            eta_min: float = 0.0,
            last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch < self.warmup_epochs:
            progress = 1.0 if self.warmup_epochs <= 1 else (
                self.last_epoch / (self.warmup_epochs - 1)
            )
            return [
                self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        cosine_epochs = self.max_epochs - self.warmup_epochs
        if cosine_epochs <= 0:
            return self.base_lrs

        progress = min(
            max(self.last_epoch - self.warmup_epochs, 0),
            cosine_epochs,
        )
        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * progress / cosine_epochs))
            for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            progress = 1.0 if self.warmup_epochs <= 1 else (
                self.last_epoch / (self.warmup_epochs - 1)
            )
            return [
                self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        cosine_epochs = self.max_epochs - self.warmup_epochs
        if cosine_epochs <= 0:
            return self.base_lrs

        progress = min(
            max(self.last_epoch - self.warmup_epochs, 0),
            cosine_epochs,
        )
        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * progress / cosine_epochs))
            for base_lr in self.base_lrs
        ]
