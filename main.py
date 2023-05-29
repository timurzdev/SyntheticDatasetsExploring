import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from dataset import CatsDataModule, CatsDataModuleSynth
from model import ResnetClassifier

if __name__ == "__main__":
    columns = ["num_synthetic_samples", "test_accuracy", "test_loss", "resnet_model"]

    df = pd.DataFrame(columns=columns)

    logger = CSVLogger(save_dir="logs")
    model_number = 101
    batch_size = 4
    trainer = pl.Trainer(
        max_epochs=100,
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
        logger=logger,
    )
    sample_nums = [i for i in range(20, 3000, 20)]
    file = open(f"log_model_{model_number}.csv", "w")
    file.write("num_samples, test_accuracy, test_loss\n")
    for num_samples in sample_nums:
        model = ResnetClassifier(2, model_number)
        data_module = CatsDataModule("./data/original/", batch_size=batch_size)
        synth_module = CatsDataModuleSynth(
            "./data/synthetic/", samples_count=num_samples, batch_size=batch_size
        )
        model.train()
        trainer.fit(model=model, datamodule=synth_module)
        model.freeze()
        trainer.test(model=model, datamodule=data_module)
        model.train()
        for param in list(model.parameters())[:-1]:
            param.requires_grad = False
        trainer.fit(model=model, datamodule=data_module)

        trainer.test(model=model, datamodule=data_module)
        file.write(f"{num_samples}, {model.test_acc}, {model.test_loss}\n")
        trainer.save_checkpoint(f"./checkpoints/{num_samples}_best.pt")
    file.close()
