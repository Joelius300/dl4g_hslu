import numpy as np
import torch
import lightning as pl

from ML.trump_selection.trump_selection import TrumpSelection
from jass.game.const import PUSH_ALT, PUSH
from jass.game.game_observation import GameObservation
from strategies.trump_strategy import TrumpStrategy


class ModelTrumpStrategy(TrumpStrategy):
    def __init__(self, checkpoint_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: pl.LightningModule = TrumpSelection.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to(device)

    def obs_to_tensor(self, obs: GameObservation) -> torch.Tensor:
        hand_one_hot = obs.hand
        fh = obs.forehand == -1

        return torch.Tensor(np.concatenate(hand_one_hot, np.array([int(fh)])))

    def y_to_trump(self, y: torch.Tensor):
        y = y.softmax(dim=-1)
        y = y.detach().cpu().numpy()

        trump = np.argmax(y)

        return trump if trump != PUSH_ALT else PUSH

    def action_trump(self, obs: GameObservation) -> int:
        with torch.no_grad():
            x = self.obs_to_tensor(obs).to(self.model.device)
            y = self.model(x)

            return self.y_to_trump(y)
