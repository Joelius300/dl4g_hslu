import argparse
import logging
import os.path
import sys

import dvc.api
import lightning as pl
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from jass_bot.ML.trump_selection.trump_graf_datamodule import TrumpGrafDataModule
from jass_bot.ML.trump_selection.trump_swisslos_datamodule import TrumpSwisslosDataModule
from jass_bot.ML.trump_selection.trump_selection import TrumpSelection

from dvclive.lightning import DVCLiveLogger

MODEL_NAME = "trump_selection"

logger = logging.getLogger("training")


def train(
    model: pl.LightningModule,
    dm: pl.LightningDataModule,
    max_epochs=100,
    early_stop_patience=5,
    log_every_n_steps=5,
    checkpoint_filename=MODEL_NAME,
    clear=False,
):
    seed_everything(42)

    # we are fine using accuracy here because the datasets we use are balanced.
    # use fixed name for checkpoint files, so they can be loaded in more easily.
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max", filename=checkpoint_filename)
    # patience is number of epochs of being worse before stopping.
    # actually it's number of val checks, but we check val once per epoch.
    # with this we slightly overtrain the model as the val_accuracy continues
    # to rise for a bit while the val_loss is already increasing again.
    # see https://stats.stackexchange.com/questions/282160/how-is-it-possible-that-validation-loss-is-increasing-while-validation-accuracy
    early_stopping = EarlyStopping(
        monitor="val_accuracy", mode="max", patience=early_stop_patience
    )
    loggers = [
        TensorBoardLogger(save_dir="lightning_logs", name=MODEL_NAME, log_graph=True),
        DVCLiveLogger(run_name=MODEL_NAME, log_model=True, resume=not clear),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
            early_stopping,
        ],
        logger=loggers,
        profiler="simple",
        log_every_n_steps=log_every_n_steps,
        precision="16-mixed",
    )

    trainer.fit(model, dm)


def get_graf_datamodule(batch_size: int, num_workers=4):
    # already pre-split into train and val with split 80/20 as seen in graf-balancing.ipynb.
    # As mentioned there as well, it's not worth it for this project but transforming graf-balancing into a script
    # and a stage in the pipeline would be much cleaner.
    return TrumpGrafDataModule(
        "data/graf-dataset-balanced/train/",
        "data/graf-dataset-balanced/val/",
        num_workers=num_workers,
        batch_size=batch_size,
    )


def get_swisslos_datamodule(batch_size: int, full: bool, test_split=0.2, num_workers=4):
    return TrumpSwisslosDataModule(
        "data/swisslos_balanced.csv" if not full else "data/swisslos_balanced_full.csv",
        # keep some test data to judge "how well the model imitates top human players" -> may not be best performance
        test_split=test_split,
        # hard code the 80/20 split here too to avoid inconsistencies
        val_split=0.2,
        num_workers=num_workers,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # for bigger projects, use argparse
    if len(sys.argv) < 2 or sys.argv[1] not in ["graf", "swisslos", "swisslos_full"]:
        logger.error("Must specify the dataset to be used ('graf', 'swisslos' or 'swisslos_full').")
        sys.exit(1)

    checkpoint_path = None
    if len(sys.argv) > 2:
        checkpoint_path = sys.argv[2]
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)

    pre_train = checkpoint_path is None
    key = "pre_train" if pre_train else "fine_tune"
    params = dvc.api.params_show()[key]
    if checkpoint_path:
        model = TrumpSelection.load_from_checkpoint(checkpoint_path)
    else:
        hidden_dim = params["hidden_dim"]
        n_layers = params["n_layers"]
        learning_rate = params["learning_rate"]
        model = TrumpSelection(hidden_dim, n_layers, learning_rate)

    batch_size = params["batch_size"]
    graf = sys.argv[1] == "graf"
    if graf:
        dm = get_graf_datamodule(batch_size)
    else:
        dm = get_swisslos_datamodule(batch_size, full=sys.argv[1].endswith("_full"), test_split=0)

    max_epochs = params["max_epochs"]
    early_stop_patience = params["early_stop_patience"]

    logger.info(
        ("Training new" if pre_train else "Fine-tuning")
        + " model on "
        + sys.argv[1]
        + " dataset"
    )

    next_checkpoint_path = MODEL_NAME + ("_pre_trained" if pre_train else "")
    train(model, dm, max_epochs, early_stop_patience, 5 if graf else 1, next_checkpoint_path, clear=pre_train)
