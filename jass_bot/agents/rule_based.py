import logging
from functools import cache

import numpy as np
from numpy.ma import MaskedArray

from heuristics.general import will_beat_trick_with_card
from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
import jass_bot.heuristics.graf as graf


class RuleBasedAgent(Agent):
    """
    Rule based agent using the Graf Heuristics.
    """
    def __init__(self):
        # log actions
        self._logger = logging.getLogger(__name__)
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        return graf.graf_trump_selection(obs)

    def action_play_card(self, obs: GameObservation) -> int:
        # card = self._get_random_card(obs)
        card = self._get_highest_winning_or_lowest_loosing_card(obs)
        self._logger.debug("Played card: {}".format(card_strings[card]))
        return card

    def _get_highest_winning_or_lowest_loosing_card(self, obs: GameObservation) -> int:
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_ids = np.flatnonzero(valid_cards)
        cards_masked = MaskedArray(valid_cards, valid_cards == 0)
        graf_scores = cards_masked * graf.get_graf_scores(obs.trump)

        highest_winning_card = None
        highest_winning_card_score = -1
        for c in valid_card_ids:
            if will_beat_trick_with_card(self._rule, c, obs):
                if graf_scores[c] > highest_winning_card_score:
                    highest_winning_card_score = graf_scores[c]
                    highest_winning_card = c

        return (
            highest_winning_card
            if highest_winning_card is not None
            else graf_scores.argmin()
        )
