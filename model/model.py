import torchvision.models as models
import lightning.pytorch as pl

import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy

from typing import Any


def get_resnet(resnet_version: int, pretrained: bool):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    return resnets[resnet_version](pretrained=pretrained, progress=True)


class ResnetClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        resnet_version: int,
        pretrained: bool,
        optimizer=optim.Adam,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.model = get_resnet(resnet_version, pretrained)
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        )
        self.lr = lr
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(linear_size, num_classes)
        self.save_hyperparameters(logger=False)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            threshold=0.0001,
            threshold_mode="abs",
        )
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            },
        )

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

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("test_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True, logger=True)
