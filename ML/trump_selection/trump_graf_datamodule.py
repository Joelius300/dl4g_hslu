import lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from jass.game.game_util import get_cards_encoded
from trump_dataset import TrumpDataset


class TrumpGrafDataModule(pl.LightningDataModule):
    # uses the generated and then balanced graf dataset. It's also already pre-split into train and val.
    # could've just generated samples on the fly but then balancing and train/val splitting would be more complex.
    def __init__(self, train_path: str, val_path: str, batch_size: int, num_workers: int):
        super().__init__()

        self.save_hyperparameters()

        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.features = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str):
        cols = [f"c{i}" for i in range(1, 10)] + ["fh", "trump"]
        train_df = pd.read_parquet(self.train_path, columns=cols)
        val_df = pd.read_parquet(self.val_path, columns=cols)
        X_train, y_train = self.to_one_hot(train_df)
        X_val, y_val = self.to_one_hot(val_df)

        self.train_dataset = TrumpDataset(X_train, y_train)
        self.val_dataset = TrumpDataset(X_val, y_val)

    def to_one_hot(self, df: pd.DataFrame):
        non_card_cols = ["fh", "trump"]
        fh_trump = df[non_card_cols]
        cards = df.drop(non_card_cols, axis=1).values
        # takes a while, but dataset isn't that big anymore, this is doable
        one_hot = np.apply_along_axis(get_cards_encoded, 1, cards)
        return (
            np.append(one_hot, np.expand_dims(fh_trump["fh"].values, axis=1), axis=1),
            fh_trump["trump"].values,
        )

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

    @staticmethod
    def collate(batch):
        assert (
            type(batch) == tuple
            and type(batch[0]) == torch.Tensor
            and type(batch[1]) == torch.Tensor
        ), "Did not get tensor from dataset, investigate and update collate"

        return batch
