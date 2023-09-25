import numpy as np

from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule
from jass.game.game_util import count_colors


def will_beat_trick_with_card(rule: GameRule, card: int, obs: GameObservation):
    """
    Returns whether the current player will beat the current trick with if they play the provided card.
    :param rule: The game rules
    :param card: The card to play
    :param obs: Current GameObservation
    :return: True if the player will beat the trick with this card (as far as we know, later players might play higher)
    """
    assert obs.nr_cards_in_trick < 4, "Number of cards must be less than 4, cannot be an already complete trick"
    tick = obs.current_trick.copy()
    tick[obs.nr_cards_in_trick] = card
    first_player = obs.trick_first_player[obs.nr_tricks]
    return rule.calc_winner(tick, first_player, obs.trump) == obs.player


def select_trump_by_number_of_cards(obs: GameObservation):
    return np.argmax(count_colors(obs.hand))
