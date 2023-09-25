import logging
from functools import cache

import numpy as np
from numpy.ma import MaskedArray

from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class RuleBasedAgent(Agent):
    def __init__(self):
        # log actions
        self._logger = logging.getLogger(__name__)
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()
        # init random number generator
        self._rng = np.random.default_rng()

    # heuristic values from Daniel Graf from 2009: "Jassen auf Basis der Spieltheorie"
    # score if the color is trump
    trump_scores = [15, 10, 7, 25, 6, 19, 5, 5, 5]
    # score if the color is not trump
    no_trump_scores = [9, 7, 5, 2, 1, 0, 0, 0, 0]
    # score if obenabe is selected (all colors)
    obenabe_scores = [14, 10, 8, 7, 5, 0, 5, 0, 0]
    # score if uneufe is selected (all colors)
    uneufe_scores = [0, 2, 1, 1, 5, 5, 7, 9, 11]
    # score threshold to exceed, otherwise push
    push_threshold = 68

    @cache
    def _get_graf_scores(self, trump: int):
        if trump == OBE_ABE:
            scores = self.obenabe_scores * 4
        elif trump == UNE_UFE:
            scores = self.uneufe_scores * 4
        else:
            scores = (
                self.no_trump_scores * trump
                + self.trump_scores
                + self.no_trump_scores * (3 - trump)
            )

        return np.array(scores)

    def action_trump(self, obs: GameObservation) -> int:
        def points_for_trump(trump: int):
            return np.sum(self._get_graf_scores(trump) * obs.hand)

        trump_scores = [points_for_trump(i) for i in range(MAX_TRUMP + 1)]
        best_trump = np.argmax(trump_scores)

        if trump_scores[best_trump] < self.push_threshold and obs.declared_trump < 0:
            return PUSH

        return best_trump

    def _select_trump_by_number_of_cards(self, obs: GameObservation):
        trump = 0
        max_number_in_color = 0
        for c in range(4):
            number_in_color = (obs.hand * color_masks[c]).sum()
            if number_in_color > max_number_in_color:
                max_number_in_color = number_in_color
                trump = c
        return trump

    def action_play_card(self, obs: GameObservation) -> int:
        # card = self._get_random_card(obs)
        card = self._get_highest_winning_or_lowest_loosing_card(obs)
        self._logger.debug("Played card: {}".format(card_strings[card]))
        return card

    def _get_random_card(self, obs: GameObservation) -> int:
        # cards are one hot encoded
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # convert to list and draw a value
        return self._rng.choice(np.flatnonzero(valid_cards))

    def _get_highest_winning_or_lowest_loosing_card(self, obs: GameObservation) -> int:
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_ids = np.flatnonzero(valid_cards)
        cards_masked = MaskedArray(valid_cards, valid_cards == 0)
        graf_scores = cards_masked * self._get_graf_scores(obs.trump)

        highest_winning_card = None
        highest_winning_card_score = -1
        for c in valid_card_ids:
            if self._will_beat_trick_with_card(c, obs):
                if graf_scores[c] > highest_winning_card_score:
                    highest_winning_card_score = graf_scores[c]
                    highest_winning_card = c

        return highest_winning_card if highest_winning_card is not None else graf_scores.argmin()

    def _will_beat_trick_with_card(self, card: int, obs: GameObservation):
        """
        Returns whether the current player will beat the current trick with if they play the provided card.
        :param card: The card to play
        :param obs: Current GameObservation
        :return: True if the player will beat the trick with this card (as far as we know, later players might play higher)
        """
        tick = obs.current_trick.copy()
        tick[obs.nr_cards_in_trick] = card
        return self._rule.calc_winner(tick, obs.trick_first_player[obs.nr_tricks], obs.trump) == obs.player
