import lightning.pytorch as pl

from model import ResnetClassifier
from dataset import CatsDataModule

if __name__ == "__main__":
    model = ResnetClassifier(2, 18)
    data_module = CatsDataModule("./data/")
    trainer = pl.Trainer(max_epochs=10, accelerator="auto", devices=1)
    trainer.fit(model=model, datamodule=data_module)
