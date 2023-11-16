import torch
from torch.utils.data import Dataset


class TrumpDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert X.shape[0] == y.shape[0], "X y dim mismatch"

    def __getitem__(self, item):
        assert False, "Inefficient __getitem__ called"

    def __getitems__(self, items):
        # linear layers and CrossEntropyLoss both need float tensors (in case of class probabilities).
        # The CrossEntropyLoss apparently is more efficient if given the class indices instead of the
        # class probabilities so no need to one-hot encode.
        return torch.FloatTensor(self.X[items]), torch.LongTensor(self.y[items])

    def __len__(self):
        return self.X.shape[0]
