import os

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import (
    random_split,
    DataLoader,
    SubsetRandomSampler,
)

from torchvision import transforms
from torchvision.datasets import ImageFolder


class CatsDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int = 16, synthetic_samples_count: int = 2000
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.synthetic_samples_count = synthetic_samples_count
        self.original_data_dir = os.path.join(data_dir, "original")
        self.synthetic_data_dir = os.path.join(data_dir, "synthetic")
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((500, 500)),
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
        indices = list(range(self.synthetic_samples_count))
        np.random.shuffle(indices)
        self.sampler = SubsetRandomSampler(indices)
        self.original_dataset = ImageFolder(
            self.original_data_dir, transform=self.transform
        )

    def setup(self, stage):
        if stage == "fit":
            (
                self.train_original_dataset,
                self.test_dataset,
                self.val_dataset,
            ) = random_split(self.original_dataset, [0.7, 0.15, 0.15])
            self.train_dataset = self.synthetic_dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=self.sampler
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
