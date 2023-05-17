import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from model import ResnetClassifier
from dataset import CatsDataModule

if __name__ == "__main__":
    model = ResnetClassifier(2, 18)
    data_module = CatsDataModule("./data/")
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="val_acc",
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode="max",
            )
        ],
    )
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)
    trainer.save_checkpoint("./checkpoints/best.pt")
