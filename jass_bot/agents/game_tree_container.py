from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.const import *
from jass.game.game_util import *
from jass.game.game_state_util import *
from jass.game.rule_schieber import RuleSchieber


@dataclass
class GameTreeNode:
    state: GameState
    """the current state of the game after that last card was played"""
    played_card: int = -1
    """the card last played (to get to this node)"""
    achieved_score: int = -1
    """the best score achievable in this subtree"""
    has_score: bool = False
    """helper to ensure setting the score"""
    children: Optional[dict[[int, GameTreeNode]]] = None
    """children of this node; one child for each playable card from this state on"""
    expanded: bool = False
    """helper to ensure population of children"""
    end_of_search: bool = False
    """signals that minimax should not continue searching from this node on"""
    total_children: int = 0
    """total children explored in this subtree"""

    def is_terminal(self):
        return self.state.nr_tricks == 9

    def is_at_end_of_trick(self):
        return self.state.nr_tricks > 0 and self.state.nr_cards_in_trick == 0


class GameTreeContainer:
    def __init__(self, rule=None):
        self._logger = logging.getLogger(__name__)

        self.root: Optional[GameTreeNode] = None
        self.cards_played_to_root: Optional[set] = None
        self._rule = rule if rule else RuleSchieber()

    def initialize_override(self, state: GameState):
        self._logger.debug("Overriding current game tree!")
        self.clear()  # just to be safe

        last_played_card = state.get_card_played(state.nr_played_cards-1) if state.nr_played_cards > 0 else -1
        self.root = GameTreeNode(state=state.clone(), played_card=last_played_card)
        self.cards_played_to_root = set()
        for ci in range(state.nr_played_cards):
            self.cards_played_to_root.add(state.get_card_played(ci))

    def initialize_if_uninitialized(self, state: GameState):
        if self.root is None:
            self.initialize_override(state)

    def clear(self):
        del self.root
        del self.cards_played_to_root
        self.root = None
        self.cards_played_to_root = None

    def find_node(self, state: GameState):
        node = self.root
        sim = None
        for trick_id in range(state.nr_tricks+1):
            trick = state.tricks[trick_id]
            for card in trick:
                if card == -1:
                    break  # this trick is done

                if card in self.cards_played_to_root:
                    continue

                if node.children is None:
                    node.children = {}

                if card not in node.children:
                    if sim is None:
                        # enough to initialize once, if one card is not in node.children,
                        # all the following won't be either so the same sim can be reused
                        sim = GameSim(self._rule)
                        sim.init_from_state(node.state)

                    sim.action_play_card(card)
                    node.children[card] = GameTreeNode(sim.state, played_card=card)

                node = node.children[card]

        return node

    def reset_end_of_search(self):
        # not sure if this is a good strategy, I could also just remove the property
        # and only rely on depth and that should also work (faster too)
        self._reset_end_of_search_for_node(self.root)

    def _reset_end_of_search_for_node(self, node: GameTreeNode):
        if node is None or not node.children:
            return

        for c in node.children.values():
            c.end_of_search = False
            self._reset_end_of_search_for_node(c)

