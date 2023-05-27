import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from model import ResnetClassifier
from dataset import CatsDataModule, CatsDataModuleSynth

if __name__ == "__main__":
    model = ResnetClassifier(2, 34)
    data_module = CatsDataModule("./data/original/")
    synth_module = CatsDataModuleSynth("./data/synthetic/", samples_count=500)
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=2,
                verbose=False,
                mode="min",
            )
        ],
    )
    model.train()
    trainer.fit(model=model, datamodule=synth_module)
    model.freeze()
    trainer.test(model=model, datamodule=data_module)
    model.train()
    trainer.fit(model=model, datamodule=data_module)
    model.freeze()
    trainer.test(model=model, datamodule=data_module)
    trainer.save_checkpoint("./checkpoints/best.pt")
