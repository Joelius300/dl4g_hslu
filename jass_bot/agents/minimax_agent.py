from __future__ import annotations

import copy
import logging
import sys
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
        card, score = self._start_minimax(state, depth=2)
        self._logger.debug(f"Playing card {card_strings[card]} after minimax to get score {score}.")
        return card

    @dataclass
    class Node:
        state: GameState  # the current state of the game after that last card was played
        played_card: int = -1  # the card last played to get to this node
        achieved_score: int = -1  # the best score achieved in this subtree (after having played that card)
        has_score: bool = False  # helper to ensure setting the score
        children: list[MiniMaxAgent.Node] = None  # one child for each playable card from this state on
        expanded: bool = False  # helper to ensure population of children
        end_of_search: bool = False  # to note that minimax should not continue searching from this node on
        total_children: int = 0

        def is_terminal(self):
            return self.state.nr_tricks == 9

        def is_at_end_of_trick(self):
            return self.state.nr_tricks > 0 and self.state.nr_cards_in_trick == 0

    # current implementation is very inefficient because it evaluates the same path for all players everytime
    # instead of having one common tree per game which all players could access so the relevant subtree could
    # be found without having to build it from the ground up again. Could be hard to implement though because
    # you'll need to keep track of where exactly you are in the game.
    def _start_minimax(self, state: GameState, depth: int) -> (int, int):
        origin_node = self.Node(state=copy.deepcopy(state))  # origin node does not have a card played, all the children will
        self._minimax(origin_node, depth_complete_tricks=depth, maximize=True, first_call=True)
        # after origin node goes through minimax, it has an achieved_score that is the maximum it can get from the
        # current position. To know how, you'll need to examine the child with the best score.
        self._logger.debug("Minimax with depth %i evaluated %i nodes", depth, origin_node.total_children)
        print(f"Minimax with depth {depth} evaluated {origin_node.total_children} nodes")
        highest_scoring_child = max(origin_node.children, key=lambda c: c.achieved_score)
        return highest_scoring_child.played_card, highest_scoring_child.achieved_score

    def _expand_node(self, node: Node):
        """
        Expands node by playing all the possible valid cards and
        adds each (unevaluated) outcome as children of this node.
        """
        if node.expanded:
            return False

        node.children = []
        valid_cards_enc = self._rule.get_valid_cards_from_state(node.state)
        valid_cards = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards_enc)
        game_sim = GameSim(self._rule)
        for card in valid_cards:
            game_sim.init_from_state(node.state)
            game_sim.action_play_card(card)
            node.children.append(self.Node(game_sim.state, played_card=card))

        node.expanded = True

        return True

    def _minimax(self, node: Node, depth_complete_tricks: int, maximize: bool, first_call=False):
        if node.end_of_search:
            assert node.has_score, "Node is end of search but has no score"
            return

        if depth_complete_tricks == 0:
            node.achieved_score = node.state.points[team[node.state.player]] * (1 if maximize else -1)
            node.has_score = True
            node.end_of_search = True
            node.total_children = 0
            return

        assert depth_complete_tricks > 0, "Depth is not positive"

        self._expand_node(node)  # ensure the children are populated (all the valid moves are played)
        assert node.expanded, "Node is not expanded after _expand_node call"

        best_child_score = sys.maxsize * (-1 if maximize else 1)
        left_depth = -1
        for child in node.children:
            if left_depth < 0:
                if not first_call and child.is_at_end_of_trick():
                    left_depth = depth_complete_tricks - 1
                else:
                    left_depth = depth_complete_tricks

            self._minimax(child, left_depth, not maximize)
            node.total_children += child.total_children + 1

            assert child.has_score, "Child does not have score after minimax"  # ensure we aren't including -1's
            if maximize:
                best_child_score = max(best_child_score, child.achieved_score)
            else:
                best_child_score = min(best_child_score, child.achieved_score)

        node.achieved_score = best_child_score
        node.has_score = True
        return
