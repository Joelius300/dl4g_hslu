from __future__ import annotations

import logging
import math
import time
import random
from typing import Callable, Optional, Iterable

import numpy as np

from heuristics import graf
from jass.agents.agent import Agent
from jass.game.const import team
from jass.game.game_rule import GameRule
from jass.game.game_sim import GameSim
from jass.game.game_state_util import *
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber

Payoffs = np.ndarray[2, np.dtype[np.float64]]


def points_div_by_max(state: GameState):
    # all points/payoffs between 0 and 1
    return state.points / np.max(state.points)


def point_div_by_norm(state: GameState):
    # sum of all payoffs = 1
    return state.points / np.linalg.norm(state.points)


# could also just use 1 and 0 for the winning and losing team, without using points directly


def UCB1(node: InformationSetMCTS.Node, total_n: int, player: int, c=1.0) -> float:
    # total_n must now be the number of times the parent was visited _AND_ node was available/valid for selection
    payoffs: float = node.W[team[player]]
    return (payoffs / node.N) + c * math.sqrt(math.log(total_n) / node.N)


def hand_consistent_with_played_card(hand: np.ndarray, played_card: int, player: int):
    return hand[player][played_card] == 0


# 144 (4*36) prime numbers for hands-hashing
# @formatter:off
# fmt: off
PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827])
# fmt: on
# @formatter:on


class InformationSetMCTS(Agent):
    class Hands:
        """Immutable but hashable container for hands of cards. Only mutate via functions that return new instances!"""

        def __init__(self, hands: np.ndarray):
            self.hands = hands

        def __len__(self):
            return self.hands.__len__()

        def __getitem__(self, key):
            return self.hands.__getitem__(key)

        def __contains__(self, key):
            return self.hands.__contains__(key)

        def __repr__(self):
            self.hands.__repr__()

        def __str__(self):
            self.hands.__str__()

        def __eq__(self, other):
            return np.array_equal(self.hands, other.hands)

        def __hash__(self):
            return np.sum(PRIMES * np.ravel(self.hands))

        def without_card(self, player: int, card: int):
            cloned = np.copy(self.hands)
            cloned[player, card] = 0
            return InformationSetMCTS.Hands(cloned)

        def has_card(self, player: int, card: int):
            return self.hands[player, card] == 1

    class InformationSet:
        def __init__(self, possible_hands: Iterable[InformationSetMCTS.Hands]):
            self.possible_hands = list(possible_hands)
            """
            List of hands where a hand is a 4x36 array that encodes in the hands of the 4 players.
            """

        def without_card(self, player: int, card: int):
            return InformationSetMCTS.InformationSet(
                # using a set, we don't have to deal with consolidation but actually not sure if that was necessary.
                {
                    hand.without_card(player, card)
                    for hand in self.possible_hands
                    if hand.has_card(player, card)
                }
            )

        def get_random_hand(self):
            return random.choice(self.possible_hands)

    class Node:
        """
        Search tree node. All the nodes in the tree are from the view of the root node player.
        """

        N: int
        """Number of simulations (random walks) started from this node."""
        W: Payoffs
        """Accumulated payoff vectors (one component for each team)."""

        cards_played_so_far: list[int]
        """All the cards played so far to get to this state."""

        information_set: InformationSetMCTS.InformationSet
        """Information set from the POV of the root node player."""

        current_trick: np.ndarray

        trump: int

        """Current game state"""
        parent: Optional[InformationSetMCTS.Node]
        """Parent node"""
        children: Optional[list[InformationSetMCTS.Node]]
        """Nodes that are possible to reach from here by playing one of the valid actions."""

        _remaining_cards: Optional[list[int]]

        def __init__(self,
                     cards_played_so_far: list[int],
                     information_set: InformationSetMCTS.InformationSet,
                     current_trick: np.ndarray,
                     nr_card_in_trick: int,
                     player: int,
                     trump: int,
                     parent: InformationSetMCTS.Node):
            self.N = 0
            self.W = np.zeros(2)
            self.cards_played_so_far = cards_played_so_far
            self.information_set = information_set
            self.current_trick = current_trick
            self.nr_cards_in_trick = nr_card_in_trick
            self.player = player
            self.trump = trump
            self.parent = parent

        @property
        def is_terminal(self):
            """Is this node at the end of a game (no more valid moves)."""
            return self.cards_played_so_far == 36

        @property
        def fully_expanded(self):
            return self._remaining_cards is not None and len(self._remaining_cards) == 0

        @property
        def has_been_sampled(self):
            return self.N > 0

        @property
        def is_root(self):
            return self.parent is None

        @property
        def is_leaf(self):
            # return self.children is None or len(self.children) == 0
            return not self.children

        @property
        def last_played_card(self):
            """Card played to get to this state"""
            if len(self.cards_played_so_far) == 0:
                return -1

            return self.cards_played_so_far[-1]

        def children_compatible_with_sample(self, sampled_hands: np.ndarray) -> bool:
            raise NotImplementedError

        def get_valid_cards_for_sampled_state(
            self, rule: GameRule, sampled_hands: InformationSetMCTS.Hands
        ):
            return rule.get_valid_cards(
                sampled_hands[self.player],
                self.current_trick,
                self.nr_cards_in_trick,
                self.trump,
            )

        def play_card(self, rule: GameRule, sampled_hands: InformationSetMCTS.Hands, parent: InformationSetMCTS.Node, card: int):
            """Plays a card and returns the node that follows from that play."""
            sim = GameSim(rule)
            sim.state.hands = sampled_hands.hands
            sim.state.player = self.player
            sim.state.trump = self.trump
            sim.state.nr_cards_in_trick = self.nr_cards_in_trick
            # write into, don't destroy references just in case
            np.copyto(sim.state.current_trick, self.current_trick)
            rule.assert_invariants(sim.state)
            sim.action_play_card(card)
            new_player = sim.state.player
            new_nr_card_in_trick = sim.state.nr_cards_in_trick
            new_current_trick = sim.state.current_trick
            new_information_set = self.information_set.without_card(
                self.player, card
            )

            return InformationSetMCTS.Node(
                card,
                new_information_set,
                new_current_trick,
                new_nr_card_in_trick,
                new_player,
                self.trump,
                parent
            )

    def __init__(
        self,
        timebudget: float,
        rule: GameRule = None,
        tree_policy: Optional[Callable[[Node, int, int], Node]] = None,
        rollout: Optional[Callable[[Node], Payoffs]] = None,
        get_payoffs: Optional[Callable[[GameState], Payoffs]] = None,
        ucb1_c_param: Optional[float] = 1.0,
    ):
        super().__init__()

        if tree_policy is None and ucb1_c_param is None:
            raise ValueError("Either provide a tree_policy or a ucb1_c_param.")

        self._logger = logging.getLogger(__name__)

        self.timebudget = timebudget
        self._root = None
        self._rule = rule if rule else RuleSchieber()
        self._tree_policy = (
            tree_policy
            if tree_policy
            else lambda node, n, p: self.UCB1_selection(node, n, p, c=ucb1_c_param)
        )
        self._rollout = rollout if rollout else self._random_walk

        default_payoff_func = points_div_by_max
        self._get_payoffs = self._get_payoffs_meta(
            get_payoffs if get_payoffs else default_payoff_func
        )

    def _get_payoffs_meta(self, get_payoffs: Callable[[GameState], Payoffs]):
        def internal_get_payoffs(state: GameState):
            assert state.nr_tricks == 9, "Tried to get payoffs from non-terminal game"
            return get_payoffs(state)

        return internal_get_payoffs

    def UCB1_selection(self, node: Node, total_n: int, player: int, c=1.0) -> Node:
        return max(node.children, key=lambda n: UCB1(n, total_n, player, c))

    def _random_walk(self, node: Node) -> Payoffs:
        if node.is_terminal:
            return self._get_payoffs(node.state)

        sim = GameSim(self._rule)
        sim.init_from_state(node.state)

        while not sim.is_done():
            card = np.random.choice(
                np.flatnonzero(self._rule.get_valid_cards_from_state(sim.state))
            )
            sim.action_play_card(card)

        return self._get_payoffs(sim.state)

    def action_trump(self, state: GameState) -> int:
        return graf.graf_trump_selection(observation_from_state(state))

    def action_play_card(self, state: GameState) -> int:
        return self.start_mcts_from_state(state, self.timebudget)

    def start_mcts_from_state(self, state: GameState, timebudget: float):
        self._root = self.Node(state, -1, None)
        # in theory, you could try to determine the last played card and parent, but it's irrelevant
        return self.start_mcts(self._root, timebudget)

    def start_mcts(self, node: Node, time_budget: float):
        """Does MCTS during a certain time_budget (in seconds) from node and returns the best card to play."""
        time_end = time.time() + time_budget

        i = 0
        while time.time() < time_end:
            i += 1
            self.mcts(node, i)

        self._logger.debug("Explored %i nodes in %.3f seconds", i, time_budget)
        return max(node.children, key=lambda n: n.N).last_played_card

    def _selection(self, node: Node, total_plays: int) -> Node:
        next_node = node
        while not next_node.is_leaf and next_node.fully_expanded:
            next_node = self._tree_policy(next_node, total_plays, node.player)

        assert (
            next_node.is_leaf or not next_node.fully_expanded
        ), "selected node is not leaf or fully_expanded"

        return next_node

    def _expansion(self, node: Node, sampled_hands: Hands) -> Node:
        assert not node.fully_expanded, "Fully expanded node in _expansion"

        # if selected node is terminal, take the payoff directly (rollout will end immediately)
        if node.is_terminal:
            return node

        # if selected has never been sampled before, do a rollout from it
        if not node.has_been_sampled:
            return node

        # if selected has already been sampled (but isn't fully expanded), select
        # random valid move and add a branch from this node. Then do a rollout from there.
        remaining_cards = node.get_valid_cards_for_sampled_state(
            self._rule, sampled_hands
        )
        sampled_card = random.choice(np.flatnonzero(remaining_cards))

        next_node = node.play_card(self._rule, sampled_hands, node, sampled_card)
        if node.children is None:
            node.children = []
        node.children.append(next_node)

        return next_node

    def _backpropagation(self, node: Node, payoffs: Payoffs):
        parent = node
        while parent is not None:
            parent.N += 1
            parent.W += payoffs
            parent = parent.parent

    def _sample_information_set(self, node: Node):
        return node.information_set.get_random_hand()

    def mcts(self, node: Node, total_plays: int):
        assert not node.is_terminal, "Started mcts from terminal node"

        sampled_hands = self._sample_information_set(node)
        next_node = self._selection(node, sampled_hands, total_plays)
        next_node = self._expansion(next_node, sampled_hands)
        payoffs = self._rollout(next_node)
        self._backpropagation(next_node, payoffs)
