import os

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import random_split, DataLoader, ConcatDataset

from torchvision import transforms
from torchvision.datasets import ImageFolder


class CatsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 16) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.original_data_dir = os.path.join(data_dir, "original")
        self.synthetic_data_dir = os.path.join(data_dir, "synthetic")
        self.transform = transforms.Compose(
            [
                transforms.ToTensor,
                transforms.Normalize(
                    (0.1307,),
                    (0.3081,),
                ),
            ]
        )

    def prepare_data(self) -> None:
        # create dataset
        self.synthetic_dataset = ImageFolder(
            self.synthetic_data_dir, transform=self.transform
        )
        self.original_dataset = ImageFolder(
            self.original_data_dir, transform=self.transform
        )

    def setup(self, stage):
        if stage == "fit":
            (
                self.train_original_dataset,
                self.test_dataset,
                self.val_dataset,
                self.predict_dataset,
            ) = random_split(self.original_dataset, [0.7, 0.1, 0.15, 0.05])
            self.train_dataset = ConcatDataset(
                [self.synthetic_dataset, self.train_original_dataset]
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)
