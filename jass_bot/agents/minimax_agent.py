import logging

import numpy as np

from heuristics.graf import graf_trump_selection
from jass.agents.agent import Agent
from jass.agents.agent_cheating import AgentCheating
from jass.game.const import *
from jass.game.game_sim import GameSim
from jass.game.game_util import *
from jass.game.game_state_util import *
from jass.game.game_observation import GameObservation
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber

class MiniMaxAgent(AgentCheating):
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._rule = RuleSchieber()

    def action_trump(self, state: GameState) -> int:
        return graf_trump_selection(observation_from_state(state))

    def action_play_card(self, state: GameState) -> int:
        card, score = self._get_points_from_action(state, tricks_left=1, maximize=True)
        return card

    # TODO nomau guet drüber nachedänkä, viläch machs dr traditioneeu wäg mit dä nodes, chasch viläch sogar es hiufestruct mache :)
    def _get_points_from_action(self, state: GameState, tricks_left: int, maximize: bool) -> (int, int):
        if tricks_left <= 0:
            assert state.nr_cards_in_trick == 0, "Should be 0 cards in trick when no trick is left to do"
            return state.trick_points[state.nr_tricks] * (1 if maximize else -1)

        valid_cards_enc = self._rule.get_valid_cards_from_state(state)
        valid_cards = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards_enc)
        card_scores = np.zeros(valid_cards_enc.shape)
        for card in valid_cards:
            game_sim = GameSim(self._rule)
            game_sim.init_from_state(state)
            game_sim.action_play_card(card)
            next_tricks_left = tricks_left
            if game_sim.state.nr_cards_in_trick == 0:
                next_tricks_left -= 1

            _best_card_after_this, card_scores[card] = self._get_points_from_action(game_sim.state, next_tricks_left, not maximize)

        best_card = card_scores.argmax() if maximize else card_scores.argmin()
        best_score = card_scores[best_card]
        self._logger.debug(f"Best card for {'max' if maximize else 'min'}imizer at tricks_left={tricks_left} = {best_card} with score {best_score}")
        assert valid_cards_enc[best_card] == 1, "Best chosen card is not a valid card!"

        return best_card, best_score
