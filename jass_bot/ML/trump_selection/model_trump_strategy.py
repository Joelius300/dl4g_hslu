import numpy as np
import torch
import lightning as pl

from jass_bot.ML.trump_selection.trump_selection import TrumpSelection
from jass.game.const import PUSH_ALT, PUSH
from jass.game.game_observation import GameObservation
from jass_bot.strategies.trump_strategy import TrumpStrategy


class ModelTrumpStrategy(TrumpStrategy):
    def __init__(self, checkpoint_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: pl.LightningModule = TrumpSelection.load_from_checkpoint(
            checkpoint_path, device
        )
        self.model.eval()

    def obs_to_tensor(self, obs: GameObservation) -> torch.Tensor:
        hand_one_hot = obs.hand
        fh = obs.forehand == -1
        fh_arr = np.array([int(fh)])
        cat = np.concatenate([hand_one_hot, fh_arr])

        return torch.tensor(cat, dtype=torch.float32)

    def y_to_trump(self, y: torch.Tensor, obs: GameObservation):
        y = y.softmax(dim=-1)
        y = y.detach().cpu().numpy()

        sorted_idx = np.argsort(y)
        best = sorted_idx[-1]
        best = best if best != PUSH_ALT else PUSH

        if best == PUSH and obs.forehand != -1:
            second_best = sorted_idx[-2]
            return second_best

        return best

    def action_trump(self, obs: GameObservation) -> int:
        with torch.no_grad():
            x = self.obs_to_tensor(obs).to(self.model.device)
            y = self.model(x)

            return self.y_to_trump(y, obs)
