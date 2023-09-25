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
    def __init__(self):
        # log actions
        self._logger = logging.getLogger(__name__)
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        def points_for_trump(trump: int):
            return np.sum(graf.get_graf_scores(trump) * obs.hand)

        trump_scores = [points_for_trump(i) for i in range(MAX_TRUMP + 1)]
        best_trump = np.argmax(trump_scores)

        if trump_scores[best_trump] < graf.push_threshold and obs.declared_trump < 0:
            return PUSH

        return best_trump

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
