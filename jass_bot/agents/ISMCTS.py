from __future__ import annotations

import logging
import math
import sys
import time
import random
from typing import Callable, Optional, Self, Union

from heuristics import graf
from jass.agents.agent import Agent
from jass.game.const import team
from jass.game.game_rule import GameRule
from jass.game.game_sim import GameSim
from jass.game.game_state_util import *
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber

Payoffs = np.ndarray[2, np.dtype[np.float64]]


def points_div_by_max(state: GameObservation | GameState):
    # all points/payoffs between 0 and 1
    return state.points / np.max(state.points)


def point_div_by_norm(state: GameObservation | GameState):
    # sum of all payoffs = 1
    return state.points / np.linalg.norm(state.points)


def UCB1(node: ISMCTS.Node, total_n: int, player: int, c=1.0) -> float:
    payoffs: float = node.W[team[player]]
    return (payoffs / node.N) + c * math.sqrt(math.log(total_n) / node.N)




class ISMCTS(Agent):
    class Node:
        """
        Search tree node. All the nodes in the tree are from the view of the root node player.
        """

        def __init__(
            self,
            known_state: GameObservation,
            last_played_card: int,
            last_player: int,
            root_player: int,
            parent: Optional[ISMCTS.Node],
        ):
            self.N = 0
            """Number of visits to this node."""
            self.parentN = 1  # initialized to 1 because it's been visited at conception, not sure that's correct
            """Number of visits to the parent node WHEN THIS NODE WAS A COMPATIBLE CHILD."""
            self.W = np.zeros(2)
            """Accumulated payoff vectors (one component for each team)."""

            self.last_played_card = last_played_card
            """The last card that was played to get to this state."""
            self.last_player = last_player
            """The player that played the last card to get to this state."""
            self.known_state = known_state
            """
            The known state of this node from the view of the root player.
            For subsequent nodes, it's derived from playing a sampled state,
            which could potentially contain uncertain information I think.
            """
            self.root_player = root_player
            """The player of the root game state. All nodes have to be this players POV."""
            assert known_state.player_view == root_player, "Observation is not root POV"

            self.parent = parent
            self.children: Optional[list[Self]] = None
            """Nodes that are possible to reach from here by playing one of the valid actions."""

            self._remaining_cards: Optional[List[int]] = None
            """The cards that can be played from this state on."""

        @property
        def is_terminal(self):
            """Is this node at the end of a game (no more valid moves)."""
            return self.known_state.nr_tricks == 9

        @property
        def fully_expanded(self):
            return self._remaining_cards is not None and len(self._remaining_cards) == 0

        @property
        def has_been_sampled(self):
            # return True  # idk man
            # return False  # just to see what happens; if this should only return true for compatible state sampling it will in practice almost always be False
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
                        rule.get_valid_cards_from_obs(self.known_state)
                    )
                )

            return self._remaining_cards

        def is_child_compatible_with_sample(self, sampled_state: GameState):
            # this child (this node as child of parent) is compatible with
            # the sample (sampled from parent) if and only if
            # the card that was played to get to this state, was in the
            # hand of the correct person in the sample.
            # a lot of other things can be assumed (here asserted)
            assert sampled_state.dealer == self.known_state.dealer
            assert sampled_state.forehand == self.known_state.forehand
            assert sampled_state.declared_trump == self.known_state.declared_trump
            assert sampled_state.trump == self.known_state.trump
            assert sampled_state.nr_played_cards + 1 == self.known_state.nr_played_cards

            return sampled_state.hands[self.last_player, self.last_played_card] == 1

        def get_children_consistent_with_sample(self, sampled_state: GameState):
            if self.children is None:
                raise ValueError("Children have not been populated yet.")

            return (
                child
                for child in self.children
                if child.is_child_compatible_with_sample(sampled_state)
            )

    def __init__(
        self,
        timebudget: float,
        rule: GameRule = None,
        tree_policy: Optional[Callable[[Node, int, int], Node]] = None,
        rollout: Optional[Callable[[Node, GameState], Payoffs]] = None,
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
            else lambda node, sample, p: self.UCB1_selection(
                node, sample, p, c=ucb1_c_param
            )
        )
        self._rollout = rollout if rollout else self._random_walk

        default_payoff_func = points_div_by_max
        self._get_payoffs = self._get_payoffs_meta(
            get_payoffs if get_payoffs else default_payoff_func
        )

    def _get_payoffs_meta(
        self, get_payoffs: Callable[[Union[GameObservation, GameState]], Payoffs]
    ):
        def internal_get_payoffs(state: GameObservation | GameState):
            assert state.nr_tricks == 9, "Tried to get payoffs from non-terminal game"
            return get_payoffs(state)

        return internal_get_payoffs

    def UCB1_selection(
        self, node: Node, sampled_state: GameState, player: int, c=1.0
    ) -> Node:
        return max(
            node.get_children_consistent_with_sample(sampled_state),
            key=lambda n: UCB1(n, n.parentN, player, c),
        )

    def _random_walk(self, node: Node, sampled_state: GameState) -> Payoffs:
        if node.is_terminal:
            return self._get_payoffs(node.known_state)

        sim = GameSim(self._rule)
        sim.init_from_state(sampled_state)

        while not sim.is_done():
            valid_cards_enc = self._rule.get_valid_cards_from_state(sim.state)
            valid_cards = np.flatnonzero(valid_cards_enc)
            assert len(valid_cards) > 0, "No valid cards but simulation not done."
            card = np.random.choice(valid_cards)
            sim.action_play_card(card)

        return self._get_payoffs(sim.state)

    def action_trump(self, observation: GameObservation) -> int:
        return graf.graf_trump_selection(observation)

    def action_play_card(self, observation: GameObservation) -> int:
        return self.start_mcts_from_obs(observation, self.timebudget)

    def start_mcts_from_obs(self, state: GameObservation, timebudget: float):
        self._root = self.Node(
            state,
            last_played_card=-1,
            last_player=-1,
            root_player=state.player,
            parent=None,
        )
        # in theory, you could try to determine the last played card and parent, but it's irrelevant
        return self.start_mcts(self._root, timebudget)

    def start_mcts(self, node: Node, time_budget: float):
        """Does MCTS during a certain time_budget (in seconds) from node and returns the best card to play."""
        time_end = time.time() + time_budget

        while time.time() < time_end:
            self.mcts(node)

        return max(node.children, key=lambda n: n.N).last_played_card

    def _sample_state(self, node: Node) -> GameState:
        played_cards = node.known_state.tricks.ravel()[
            : node.known_state.nr_played_cards
        ]
        played_cards = set(played_cards)
        hand_cards = set(np.flatnonzero(node.known_state.hand))
        remaining_cards = ALL_CARDS - played_cards - hand_cards
        remaining_cards = np.array(list(remaining_cards))
        np.random.shuffle(remaining_cards)
        distributed_hands = np.array_split(remaining_cards, 3)

        hands = np.zeros(shape=[4, 36], dtype=np.int32)
        hands[node.root_player, :] = node.known_state.hand

        skip_correction = 0
        start_player = node.known_state.trick_first_player[node.known_state.nr_tricks]

        p = start_player
        for i in range(4):
            if p == node.root_player:
                # must shift after root player to not override their hand
                skip_correction = 1
                p = next_player[p]
                continue

            # index from behind because the size of the chunks is descending.
            # this way the first player (start player), will get the first
            # of the smaller chunks, then the next, etc.
            hand_indices = distributed_hands[-((i + 1) - skip_correction)]
            hands[p, hand_indices] = 1

            # for the players that have already played (i < nr_card_in_trick)
            # the number of cards should be floor(remaining_cards / 3)
            # and for those who haven't (i >= nr_cards_in_trick) it should be 1 more
            if node.known_state.nr_cards_in_trick > 0:
                assert len(hand_indices) == len(remaining_cards) // 3 + (
                    0 if i < node.known_state.nr_cards_in_trick else 1
                )
            else:
                # if there are 0 in the trick, everyone should have floor(remaining / 3).
                assert len(hand_indices) == len(remaining_cards) // 3

            p = next_player[p]

        return state_from_observation(node.known_state, hands)

    def _selection(self, node: Node, sampled_state: GameState) -> Node:
        next_node = node
        while not next_node.is_leaf and next_node.fully_expanded:
            next_node = self._tree_policy(
                next_node, sampled_state, node.known_state.player
            )

        assert (
            next_node.is_leaf or not next_node.fully_expanded
        ), "selected node is not leaf or fully_expanded"

        return next_node

    def _expansion(self, node: Node, sampled_state: GameState) -> Node:
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
        game_sim.init_from_state(sampled_state)
        game_sim.action_play_card(sampled_card)

        next_node = ISMCTS.Node(
            observation_from_state(
                # removes the information we can't know, like
                # the hands of the others. However, it still
                # leaks information because it keeps who won
                # the last trick -> order of players.
                # may not be an issue because incompatible ones are filtered anyway.
                game_sim.state,
                node.root_player,
            ),
            sampled_card,
            node.known_state.player,
            node.root_player,  # root player stays
            node,
        )
        if node.children is None:
            node.children = []
        node.children.append(next_node)

        return next_node

    def _backpropagation(self, node: Node, payoffs: Payoffs, sampled_state: GameState):
        parent = node
        while parent is not None:
            parent.N += 1
            parent.W += payoffs

            if parent.children:
                for child in parent.get_children_consistent_with_sample(sampled_state):
                    # children need to know how many times their parents were simulated
                    # and them being compatible. Rooms for performance improvements, surely.
                    child.parentN += 1

            parent = parent.parent

    def mcts(self, node: Node):
        assert not node.is_terminal, "Started mcts from terminal node"
        sampled_state = self._sample_state(node)

        self._assert_sample_validity(node, sampled_state)

        next_node = self._selection(node, sampled_state)
        next_node = self._expansion(next_node, sampled_state)
        payoffs = self._rollout(next_node, sampled_state)
        self._backpropagation(next_node, payoffs, sampled_state)

    def _assert_sample_validity(self, node: ISMCTS.Node, sampled_state: GameState):
        self._rule.assert_invariants(sampled_state)
        assert node.known_state.forehand == sampled_state.forehand
        assert node.known_state.declared_trump == sampled_state.declared_trump
        assert node.known_state.dealer == sampled_state.dealer
        assert node.known_state.trump == sampled_state.trump
        assert node.known_state.player == sampled_state.player
        assert node.known_state.nr_played_cards == sampled_state.nr_played_cards
        assert node.known_state.nr_tricks == sampled_state.nr_tricks
        assert node.known_state.nr_cards_in_trick == sampled_state.nr_cards_in_trick
        assert np.array_equal(node.known_state.tricks, sampled_state.tricks)
        assert np.array_equal(node.known_state.trick_winner, sampled_state.trick_winner)
        assert np.array_equal(node.known_state.trick_points, sampled_state.trick_points)
        assert np.array_equal(node.known_state.points, sampled_state.points)
        assert np.array_equal(
            node.known_state.trick_first_player, sampled_state.trick_first_player
        )
        assert np.array_equal(
            node.known_state.hand, sampled_state.hands[node.root_player]
        )
        assert node.known_state == observation_from_state(
            sampled_state, node.root_player
        )
