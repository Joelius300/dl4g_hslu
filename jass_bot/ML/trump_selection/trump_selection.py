import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


INPUT_DIM = 36 + 1  # all cards + forehand
N_CLASSES = 7  # all trumps + push


class TrumpSelection(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        learning_rate: float,
        dropout_rate: float = 0,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Experiments to do
        # - Dropout
        # - Residual connections
        #   - First have a linear layer which projects the hand into some space with some features already (proj_dim)
        #   - Concat that projection with the original hand to get (proj_dim + input_dim)
        #   - Feed that to a linear layer which downprojects to proj_dim, then activation function
        #   - Repeat last two
        #   - Feed into classifier, again concatenating with the original hand first
        self.ll = nn.ModuleList(
            [nn.Linear(INPUT_DIM, hidden_dim)]
            + [nn.Linear(hidden_dim + INPUT_DIM, hidden_dim) for _ in range(n_layers - 1)]
        )

        self.classifier = nn.Linear(hidden_dim + INPUT_DIM, N_CLASSES)
        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = nn.ModuleDict(
            dict(
                accuracy=torchmetrics.Accuracy("multiclass", num_classes=N_CLASSES),
                precision=torchmetrics.Precision("multiclass", num_classes=N_CLASSES),
                recall=torchmetrics.Recall("multiclass", num_classes=N_CLASSES),
                f1=torchmetrics.F1Score("multiclass", num_classes=N_CLASSES),
            )
        )

        self.learning_rate = learning_rate

        self.example_input_array = torch.tensor(
            [1] * 9 + [0] * (INPUT_DIM - 9), dtype=torch.float
        )

    def forward(self, x: torch.Tensor):
        orig = x
        for l in self.ll:
            x = self.dropout(x)
            x = l(x)
            x = torch.concat([x, orig], dim=-1)  # always give the layer access to the initial input
            x = F.relu(x)
        x = self.classifier(x)

        # no softmax here because CrossEntropyLoss does that internally for better numerical stability
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, _batch_idx):
        return self.step("train_", batch)

    def validation_step(self, batch, _batch_idx):
        return self.step("val_", batch)

    def test_step(self, batch, _batch_idx):
        return self.step("test_", batch)

    def step(self, prefix, batch):
        X, y = batch
        predictions = self(X)
        loss = self.criterion(predictions, y)
        self.log(prefix + "loss", loss)

        # remember, prediction is still the logits.
        # many of these metrics should be able to handle that
        # but for efficiency and to be sure, let's do the softmax ourselves.
        predictions = F.softmax(predictions, dim=-1)
        self._log_and_update_metrics(prefix, predictions, y)

        return loss

    def _log_and_update_metrics(self, prefix, prediction, y):
        for name, metric in self.metrics.items():
            metric(prediction, y)
            self.log(prefix + name, metric)
