from __future__ import annotations

import math
import time
from typing import Callable, Optional

from heuristics import graf
from jass.agents.agent_cheating import AgentCheating
from jass.game.const import team
from jass.game.game_rule import GameRule
from jass.game.game_sim import GameSim
from jass.game.game_state_util import *
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber

Payoffs = np.ndarray[2, np.dtype[np.float64]]


# Payoffs normalized 0,1
class CheatingMCTS(AgentCheating):
    # @dataclass
    class Node:
        N: int
        """Number of simulations (random walks) started from this node."""
        W: Payoffs
        """Accumulated payoff vectors (one component for each player)."""

        last_played_card: int
        """Card played to get to this state"""
        state: GameState
        """Current game state"""
        parent: Optional[CheatingMCTS.Node]
        """Parent node"""
        children: Optional[list[CheatingMCTS.Node]]
        """Game states that are possible to reach from here by playing one of the valid actions."""
        # _valid_actions: Optional[list[int]]

        def __init__(self, state: GameState, last_played_card: int):
            self.state = state
            self.last_played_card = last_played_card
            self.N = 0
            self.W = np.zeros(2)
            self.parent = None
            self.children = None

        @property
        def is_terminal(self):
            """Is this node at the end of a game (no more valid moves)."""
            return self.state.nr_tricks == 9

        @property
        def fully_expanded(self):
            return self.children is not None

        @property
        def has_been_sampled(self):
            return self.N > 0

        @property
        def is_root(self):
            return self.parent is None

        # def valid_cards(self, rule: GameRule):
        #     # assumption: always called with the same rule!
        #     if not self._valid_actions:
        #         self._valid_actions = convert_one_hot_encoded_cards_to_int_encoded_list(
        #             rule.get_valid_cards_from_state(self.state)
        #         )
        #
        #     return self._valid_actions

    def __init__(
        self,
        timebudget: float,
        rule: GameRule = None,
        tree_policy: Optional[Callable[[Node, int, int], Node]] = None,
        rollout: Optional[Callable[[Node], Payoffs]] = None,
        ucb1_c_param: Optional[float] = 1.0,
    ):
        super().__init__()

        if tree_policy is None and ucb1_c_param is None:
            raise ValueError("Either provide a tree_policy or a ucb1_c_param.")

        self.timebudget = timebudget
        self._root = None
        self._rule = rule if rule else RuleSchieber()
        self._tree_policy = (
            tree_policy
            if tree_policy
            else lambda node, n, p: self.UCB1_selection(node, n, p, c=ucb1_c_param)
        )
        self._rollout = rollout if rollout else self._random_walk

    def UCB1(self, node: Node, total_n: int, player: int, c=1.0) -> float:
        payoff: float = node.W[team[player]]  # no clue why this is a type-mismatch
        return (payoff / node.N) + c * math.sqrt(math.log(total_n) / node.N)

    def UCB1_selection(self, node: Node, total_n: int, player: int, c=1.0) -> Node:
        return max(node.children, key=lambda n: self.UCB1(n, total_n, player, c))

    def _random_walk(self, node: Node) -> Payoffs:
        sim = GameSim(self._rule)
        sim.init_from_state(node.state)

        while not sim.is_done():
            card = np.random.choice(
                np.flatnonzero(self._rule.get_valid_cards_from_state(sim.state))
            )
            sim.action_play_card(card)

        # sum of all payoffs = 1
        # return sim.state.points / np.linalg.norm(sim.state.points)

        # could also just use 1 and 0 for the winning and losing team, without using points directly

        # all points/payoffs between 0 and 1
        return sim.state.points / np.max(sim.state.points)

    def action_trump(self, state: GameState) -> int:
        return graf.graf_trump_selection(observation_from_state(state))

    def action_play_card(self, state: GameState) -> int:
        return self.start_mcts_from_state(state, self.timebudget)

    def _expand_node(self, node: Node):
        """
        Expands node by playing all the possible valid cards and
        adds each (unevaluated) outcome as children of this node.
        """
        assert node.children is None, "Cannot expand node with children."

        node.children = []
        valid_cards_enc = self._rule.get_valid_cards_from_state(node.state)
        valid_cards = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards_enc)
        game_sim = GameSim(self._rule)
        for card in valid_cards:
            game_sim.init_from_state(node.state)
            game_sim.action_play_card(card)
            node.children.append(CheatingMCTS.Node(game_sim.state, card))

    def start_mcts_from_state(self, state: GameState, timebudget: float):
        self._root = self.Node(state, -1)
        # in theory, you could try to determine the last played card but it's irrelevant
        return self.start_mcts(self._root, timebudget)

    def start_mcts(self, node: Node, time_budget: float):
        """Does MCTS during a certain time_budget (in seconds) from node and returns the best card to play."""
        time_end = time.time() + time_budget

        i = 1
        while time.time() < time_end:
            self.mcts(node, i)
            i += 1

        return max(node.children, key=lambda n: n.N).last_played_card

    def mcts(self, node: Node, total_plays: int):
        # selection
        next_node = node
        while next_node.fully_expanded or next_node.is_terminal:
            next_node = self._tree_policy(next_node, total_plays, node.state.player)

        # expansion
        if next_node.has_been_sampled:
            self._expand_node(next_node)
            next_node = next_node.children[0]  # not terminal, there must be children
            # TODO do you have to do a rollout from _all_ the children? otherwise how to ensure that N>=1 everywhere

        # rollout
        payoffs = self._rollout(next_node)

        # backpropagation
        parent = next_node
        while parent is not None:
            parent.N += 1
            parent.W += payoffs
            parent = next_node.parent
