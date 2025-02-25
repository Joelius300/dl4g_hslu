from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np

from jass_bot.agents.game_tree_container import GameTreeContainer, GameTreeNode
from jass_bot.heuristics.graf import graf_trump_selection
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
    def __init__(self, tree: GameTreeContainer, depth=1):
        self._logger = logging.getLogger(__name__)
        self._rule = RuleSchieber()
        self._depth = depth
        self._tree = tree

    def setup(self, game_id: Optional[int] = None):
        self._tree.clear()  # this only happens at the start of the game,
        # so we can clear the entire tree and open it for initialization
        # not in the original library, maybe not usable there

    def action_trump(self, state: GameState) -> int:
        return graf_trump_selection(observation_from_state(state))

    def action_play_card(self, state: GameState) -> int:
        card, score = self._start_minimax(state, self._depth)
        self._logger.debug(f"Playing card {card_strings[card]} after minimax to get score {score}.")
        return card

    def _start_minimax(self, state: GameState, depth: int) -> (int, int):
        # initialize tree at the first card action from our side, meaning some
        # other players could have played already. Therefore, the root may not be the very initial state.
        self._tree.initialize_if_uninitialized(state)

        origin_node = self._tree.find_node(state)  # retrieve game tree node without calculating it everytime

        self._minimax(origin_node, depth_complete_tricks=depth, maximize=True, first_call=True)
        # after origin node goes through minimax, it has an achieved_score that is the maximum it can get from the
        # current position. To know how, you'll need to examine the child with the best score.
        highest_scoring_child = max(origin_node.children.values(), key=lambda c: c.achievable_score)

        assert state == origin_node.state, "State of origin node is not in sync with original state anymore"
        return highest_scoring_child.played_card, highest_scoring_child.achievable_score

    def _expand_node(self, node: GameTreeNode):
        """
        Expands node by playing all the possible valid cards and
        adds each (unevaluated) outcome as children of this node.
        """
        if node.children is not None:
            return False

        node.children = {}
        valid_cards_enc = self._rule.get_valid_cards_from_state(node.state)
        valid_cards = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards_enc)
        game_sim = GameSim(self._rule)
        for card in valid_cards:
            game_sim.init_from_state(node.state)
            game_sim.action_play_card(card)
            node.children[card] = GameTreeNode(game_sim.state, played_card=card)

        return True

    def _minimax(self, node: GameTreeNode, depth_complete_tricks: int, maximize: bool, first_call=False):
        if depth_complete_tricks == 0:
            node.achievable_score = node.state.points[team[node.state.player]] * (1 if maximize else -1)
            return

        assert depth_complete_tricks > 0, "Depth is not positive"

        self._expand_node(node)  # ensure the children are populated (all the valid moves are played)

        best_child_score = sys.maxsize * (-1 if maximize else 1)
        left_depth = -1
        for child in node.children.values():
            if left_depth < 0:
                if not first_call and child.is_at_end_of_trick():
                    left_depth = depth_complete_tricks - 1
                else:
                    left_depth = depth_complete_tricks

            # the assumption that maximize and minimize always switch like tic-tac-toe for example is wrong
            # because of the transition from one trick to the other where the last winner is first
            self._minimax(child, left_depth, maximize if same_team[node.state.player, child.state.player] else not maximize)

            if maximize:
                best_child_score = max(best_child_score, child.achievable_score)
            else:
                best_child_score = min(best_child_score, child.achievable_score)

        node.achievable_score = best_child_score
