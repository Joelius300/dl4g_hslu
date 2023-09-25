import copy
import logging
from dataclasses import dataclass
from typing import NamedTuple

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
        card, score = self._start_minimax(state, depth=4)
        return card

    @dataclass
    class Node:
        state: GameState
        best_score: int = -1
        best_card: int = -1

    def _start_minimax(self, state: GameState, depth: int) -> (int, int):
        origin_node = self.Node(state=state)
        self._explore_node(origin_node, depth, maximize=True)
        return origin_node.best_card, origin_node.best_score

    def _explore_node(self, node: Node, depth: int, maximize: bool):
        state = node.state
        game_sim = GameSim(self._rule)
        game_sim.init_from_state(state)
        if depth <= 0 or game_sim.is_done():
            # assert state.nr_cards_in_trick == 0, "Should be 0 cards in trick when no trick is left to do (depth=0)"
            node.best_score = state.trick_points[state.nr_tricks-1] * (1 if maximize else -1)
            return

        valid_cards_enc = self._rule.get_valid_cards_from_state(state)
        valid_cards = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards_enc)

        best_node = None
        best_card = -1
        for card in valid_cards:
            game_sim.init_from_state(state)
            game_sim.action_play_card(card)
            next_node = self.Node(game_sim.state)
            self._explore_node(next_node, depth-1, not maximize)
            if (best_node is None or
                    (maximize and next_node.best_score > best_node.best_score) or
                    (not maximize and next_node.best_score < best_node.best_score)):
                best_node = next_node
                best_card = card

        node.best_score = best_node.best_score
        node.best_card = best_card
        self._logger.debug(f"Best card for {'max' if maximize else 'min'}imizer at depth={depth} = {card_strings[best_card]} with score {node.best_score}")
        assert valid_cards_enc[best_card] == 1, "Best chosen card is not a valid card!"
