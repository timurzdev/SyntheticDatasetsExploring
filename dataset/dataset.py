import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class CatsDataModuleSynth(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.samples_count = 1000
        self.path = path
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def prepare_data(self) -> None:
        # create dataset
        self.synthetic_dataset = ImageFolder(self.path, transform=self.transform)
        if self.samples_count != 0:
            indices = list(range(self.samples_count))
            np.random.shuffle(indices)
            self.sampler = SubsetRandomSampler(indices)

    def setup(self, stage):
        if stage == "fit":
            (
                self.train_dataset,
                self.val_dataset,
            ) = random_split(self.synthetic_dataset, [0.7, 0.3])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.samples_count != 0:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.sampler,
                num_workers=4,
            )
        else:
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size, num_workers=4
            )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


class CatsDataModule(pl.LightningDataModule):
    def __init__(self, path: str, batch_size: int = 16) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.path = path
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def prepare_data(self) -> None:
        self.dataset = ImageFolder(self.path, transform=self.transform)

        (
            self.train_dataset,
            self.val_dataset,
        ) = random_split(self.dataset, [0.8, 0.2])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


class CatsDataModuleTest(pl.LightningDataModule):
    def __init__(self, path: str, batch_size: int = 16) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.path = path
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def prepare_data(self) -> None:
        self.dataset = ImageFolder(self.path, transform=self.transform)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4)
