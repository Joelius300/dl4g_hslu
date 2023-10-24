from __future__ import annotations

import logging
import math
import time
import random
from typing import Callable, Optional

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


def UCB1(node: ISMCTS.Node, total_n: int, player: int, c=1.0) -> float:
    payoffs: float = node.W[team[player]]
    return (payoffs / node.N) + c * math.sqrt(math.log(total_n) / node.N)


class ISMCTS(Agent):
    # @dataclass
    class Node:
        N: int
        """Number of simulations (random walks) started from this node."""
        W: Payoffs
        """Accumulated payoff vectors (one component for each team)."""

        last_played_card: int
        """Card played to get to this state"""
        state: GameState
        """Current game state"""
        parent: Optional[ISMCTS.Node]
        """Parent node"""
        children: Optional[list[ISMCTS.Node]]
        """Game states that are possible to reach from here by playing one of the valid actions."""

        _remaining_cards: Optional[list[int]]

        def __init__(
            self,
            state: GameState,
            last_played_card: int,
            parent: Optional[ISMCTS.Node],
        ):
            self.state = state
            self.last_played_card = last_played_card
            self.N = 0
            self.W = np.zeros(2)
            self.parent = parent
            self.children = None
            self._remaining_cards = None

        @property
        def is_terminal(self):
            """Is this node at the end of a game (no more valid moves)."""
            return self.state.nr_tricks == 9

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

        def remaining_cards(self, rule: GameRule) -> list[int]:
            """Gets the valid cards that have never been played from this node."""
            # assumption: always called with the same rule!
            if self._remaining_cards is None:
                self._remaining_cards = (
                    convert_one_hot_encoded_cards_to_int_encoded_list(
                        rule.get_valid_cards_from_state(self.state)
                    )
                )

            return self._remaining_cards

    def __init__(
        self,
        timebudget: float,
        rule: GameRule = None,
        tree_policy: Optional[Callable[[Node, int, int], Node]] = None,
        rollout: Optional[Callable[[Node], Payoffs]] = None,
        get_payoffs: Optional[Callable[[GameState], Payoffs]] = None,
        ucb1_c_param: Optional[float] = math.sqrt(2),
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

    def action_trump(self, observation: GameObservation) -> int:
        return graf.graf_trump_selection(observation)

    def action_play_card(self, observation: GameObservation) -> int:
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
            next_node = self._tree_policy(next_node, total_plays, node.state.player)

        assert (
            next_node.is_leaf or not next_node.fully_expanded
        ), "selected node is not leaf or fully_expanded"

        return next_node

    def _expansion(self, node: Node) -> Node:
        assert not node.fully_expanded, "Fully expanded node in _expansion"

        # if selected node is terminal, take the payoff directly (rollout will end immediately)
        if node.is_terminal:
            return node

        # if selected has never been sampled before, do a rollout from it
        if not node.has_been_sampled:
            return node

        # if selected has already been sampled (but isn't fully expanded), select
        # random valid move and add a branch from this node. Then do a rollout from there.
        remaining_cards = node.remaining_cards(self._rule)
        if len(remaining_cards) == 1:
            i = 0
        else:
            i = random.randint(0, len(remaining_cards) - 1)

        sampled_card = remaining_cards.pop(i)
        game_sim = GameSim(self._rule)
        game_sim.init_from_state(node.state)
        game_sim.action_play_card(sampled_card)
        next_node = ISMCTS.Node(game_sim.state, sampled_card, node)
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

    def mcts(self, node: Node, total_plays: int):
        assert not node.is_terminal, "Started mcts from terminal node"
        next_node = self._selection(node, total_plays)
        next_node = self._expansion(next_node)
        payoffs = self._rollout(next_node)
        self._backpropagation(next_node, payoffs)
