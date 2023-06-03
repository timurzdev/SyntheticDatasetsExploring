import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from dataset import CatsDataModule, CatsDataModuleSynth, CatsDataModuleTest
from model import ResnetClassifier


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_synth(
    num_samples: int,
    model_number: int,
    batch_size: int,
    logger_name: str,
    pretrained: bool = False,
):
    logger = WandbLogger(logger_name)
    trainer = pl.Trainer(
        max_epochs=500,
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
    model = ResnetClassifier(2, model_number, lr=1e-6)
    data_module = CatsDataModuleTest("./test/", batch_size=batch_size)
    synth_module = CatsDataModuleSynth(
        "./data/synthetic/", samples_count=num_samples, batch_size=batch_size
    )
    model.train()
    trainer.fit(model=model, datamodule=synth_module)
    model.eval()
    trainer.test(model=model, datamodule=data_module)
    trainer.save_checkpoint(f"./checkpoints/{logger_name}_best.ckpt")


def train(
    model_number: int,
    batch_size: int,
    logger_name: str,
    mixed_train: bool,
    pretrained: bool = False,
):
    logger = WandbLogger(logger_name)
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=5,
                verbose=False,
                mode="min",
            )
        ],
        logger=logger,
    )
    model = ResnetClassifier(2, model_number, lr=1e-2)
    data_module = CatsDataModuleTest("./test/", batch_size=batch_size)
    train_module = CatsDataModule("./data/original/", batch_size=batch_size)
    model.train()
    if mixed_train:
        # training on synthetic data first
        synth_module = CatsDataModuleSynth("./data/synthetic/", batch_size=batch_size)
        trainer.fit(model=model, datamodule=synth_module)
        # # freezing backbone
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False

    trainer.fit(model=model, datamodule=train_module)
    model.eval()
    trainer.test(model=model, datamodule=data_module)
    trainer.save_checkpoint(f"./checkpoints/{logger_name}_best.ckpt")


def predict(model_number: int, batch_size: int, logger_name: str):
    logger = WandbLogger(logger_name)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
    )
    model = ResnetClassifier(2, model_number, lr=1e-5)
    data_module = CatsDataModule("./test/", batch_size=batch_size)
    trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path=f"./checkpoints/{logger_name}_best.ckpt",
    )


if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixed-train", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "-t", "--train", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "-p", "--predict", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("-n", "--num_samples", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-l", "--logger_name", type=str, required=True)
    parser.add_argument("-m", "--model", type=int, default=101)
    args = parser.parse_args()
    print(args)
    if args.train:
        train(
            model_number=args.model,
            batch_size=args.batch_size,
            logger_name=args.logger_name,
            mixed_train=args.mixed_train,
        )
    if args.predict:
        predict(
            model_number=args.model,
            batch_size=args.batch_size,
            logger_name=args.logger_name,
        )
