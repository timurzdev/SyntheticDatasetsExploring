from typing import Any
import torchvision.models as models
import lightning.pytorch as pl

import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy


def get_resnet(resnet_version: int):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    return resnets[resnet_version](pretrained=False, progress=True)


class ResnetClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        resnet_version: int,
        optimizer=optim.Adam,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.model = get_resnet(resnet_version)
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        )
        self.lr = lr
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(linear_size, num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Any:
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch
        preds = self(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
