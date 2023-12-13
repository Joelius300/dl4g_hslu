import lightning as pl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from jass.game.const import card_strings
from jass_bot.ML.trump_selection.trump_dataset import TrumpDataset


class TrumpSwisslosDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        test_split: float,
        val_split: float,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.csv_path = csv_path
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = None
        self.promising_users = None
        self.features = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        self.features = np.append(card_strings, ["FH"])
        self.data = pd.read_csv(self.csv_path)

        X = self.data[self.features].values
        fh = self.data["FH"].values
        y = self.data["trump"].values

        # we need stratification, otherwise torch's random_split would work too.
        # Stratify with a combination of fh and trump: 00,01,02,03,04,05,10,11,12,13,14,15,16
        if self.test_split > 0:
            X_train, X_test, y_train, y_test, fh_train, _ = train_test_split(
                X, y, fh, test_size=self.test_split, stratify=(fh * 10 + y), random_state=42
            )
        else:
            X_train, y_train, fh_train = X, y, fh
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.val_split,
            stratify=(fh_train * 10 + y_train),
            random_state=42,
        )

        self.train_dataset = TrumpDataset(X_train, y_train)
        self.val_dataset = TrumpDataset(X_val, y_val)
        self.test_dataset = TrumpDataset(X_test, y_test) if self.test_split > 0 else None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise Exception("No test dataloader because test_split was not positive.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )

    @staticmethod
    def collate(batch):
        # the default collate thinks __getitems__ returns a list of tuples that need to be stacked to get the batched
        # values just like you would need to if you invoked __getitem__ multiple times and tried to batch that together.
        # however, __getitems__ already returns the tensor with the items stacked so no need for additional processing.
        # see https://pytorch.org/docs/stable/data.html#torch.utils.data._utils.collate.collate potentially
        assert (
            type(batch) == tuple
            and type(batch[0]) == torch.Tensor
            and type(batch[1]) == torch.Tensor
        ), "Did not get tensor from dataset, investigate and update collate"

        return batch
