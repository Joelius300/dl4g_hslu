import numpy as np

from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule


def get_random_valid_card(rule: GameRule, obs: GameObservation) -> int:
    # cards are one hot encoded
    valid_cards = rule.get_valid_cards_from_obs(obs)
    # convert to list and draw a value
    return np.random.choice(np.flatnonzero(valid_cards))
