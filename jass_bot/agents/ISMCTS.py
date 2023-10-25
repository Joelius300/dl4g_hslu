from __future__ import annotations

import logging
import math
import time
from typing import Callable, Optional, Self, Union, Tuple

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


def UCB1(node: ISMCTS.Node, total_n: int, player: int, c=1.0) -> float:
    payoffs: float = node.W[team[player]]
    return (payoffs / node.N) + c * math.sqrt(math.log(total_n) / node.N)


ALL_CARDS = frozenset(range(36))


def _get_remaining_cards_in_play_from_obs(obs: GameObservation):
    """Returns all cards that have not yet been played and aren't in the players hand."""
    played_cards = obs.tricks.ravel()[: obs.nr_played_cards]
    played_cards = set(played_cards)
    hand_cards = set(np.flatnonzero(obs.hand))
    remaining_cards = ALL_CARDS - played_cards - hand_cards
    remaining_cards = list(remaining_cards)
    return remaining_cards


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

            num_card_in_hands_at_start_of_trick = 36 - (
                known_state.nr_played_cards - known_state.nr_cards_in_trick
            )
            assert (
                num_card_in_hands_at_start_of_trick % 4 == 0
            ), "num_card_in_hands_at_start_of_trick is not divisible by 4"
            cards_in_hand = num_cards_in_hand(known_state.hand)
            assert cards_in_hand == num_card_in_hands_at_start_of_trick / 4 or (
                cards_in_hand == (num_card_in_hands_at_start_of_trick / 4) - 1
                and known_state.nr_cards_in_trick > 0
            ), "Hand has invalid number of cards"

            self.root_player = root_player
            """The player of the root game state. All nodes have to be this players POV."""
            assert known_state.player_view == root_player, "Observation is not root POV"

            self.parent = parent

            if self.parent:
                assert (
                    self.parent.known_state.nr_played_cards
                    == self.known_state.nr_played_cards - 1
                ), "Parent does not have one less card played than child."

            self.children: Optional[list[Self]] = None
            """Nodes that are possible to reach from here by playing one of the valid actions."""

            self._last_sampled_state: Optional[GameState] = None
            """
            The last sampled that with that was handled with this node.
            LOOOOTS of potential for bugs when this is not handled/set/reset properly.
            Only set in _selection and _expansions and only use in _backprop.
            Reset after that it's None outside of the time between _selection and _backprop.
            """

            self._remaining_cards: np.array = None
            """One-hot encoded set of the cards that have not been explored from this position."""
            self._played_cards: np.array = np.zeros(36, dtype=int)
            """
            One-hot encoded set of the cards that have been explored from this position.
            For each card, children should contain a corresponding child.
            """

        @property
        def is_terminal(self):
            """Is this node at the end of a game (no more valid moves)."""
            return self.known_state.nr_tricks == 9

        @property
        def last_sampled_state(self):
            return self._last_sampled_state

        @last_sampled_state.setter
        def last_sampled_state(self, value: GameState):
            assert value is None or (
                value.nr_played_cards == self.known_state.nr_played_cards
                and np.array_equal(value.player, self.known_state.player)
                and np.array_equal(value.current_trick, self.known_state.current_trick)
                and np.array_equal(value.tricks, self.known_state.tricks)
            ), "Tried setting a last sample that doesn't align with known state."
            assert (value is None) != (
                self._last_sampled_state is None
            ), "Tried to unset or set a value when it was[nt] None"

            # to avoid any nastiness with mutable game states (which happen very easily, trust me)
            # clone the state no matter where it comes from.
            self._last_sampled_state = value.clone() if value is not None else None

        def fully_expanded_for(self, rule: GameRule, sampled_state: GameState):
            """
            Returns if this node has been fully expanded for this sampled state and this rule (True),
            or if there are still unexplored valid moves left to play.
            """
            return (
                self._remaining_cards is not None
                and num_cards_in_hand(self._get_valid_cards(rule, sampled_state)) == 0
            )

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
            assert self.parent.last_sampled_state == sampled_state
            assert self.known_state.nr_played_cards == sampled_state.nr_played_cards + 1

            return sampled_state.hands[self.last_player, self.last_played_card] == 1

        def get_children_consistent_with_sample(self, sampled_state: GameState):
            if self.children is None:
                raise ValueError("Children have not been populated yet.")

            return [
                child
                for child in self.children
                if child.is_child_compatible_with_sample(sampled_state)
            ]

        def _get_valid_cards(self, rule: GameRule, sampled_state: GameState):
            assert (
                sampled_state.player == self.known_state.player
            ), "Sampled state has different player's turn"
            assert (
                sampled_state.nr_played_cards == self.known_state.nr_played_cards
            ), "Sample and known state out of sync"
            if self._remaining_cards is None:
                if self.known_state.player == self.root_player:
                    # we have perfect information
                    self._remaining_cards = rule.get_valid_cards_from_obs(self.known_state)
                else:
                    # imperfect information, just take all remaining cards
                    # the valid ones are filtered with every query
                    cards = _get_remaining_cards_in_play_from_obs(self.known_state)
                    self._remaining_cards = get_cards_encoded(cards)

            assert self._remaining_cards is not None, "No remaining cards"

            valid_cards_in_sample = rule.get_valid_cards_from_state(sampled_state)
            valid_remaining_cards = self._remaining_cards & valid_cards_in_sample
            assert np.array_equal(
                valid_remaining_cards.astype(bool),
                (self._remaining_cards.astype(bool) & valid_cards_in_sample.astype(bool)),
            ), "Binary AND does not work on int array"
            assert len(valid_remaining_cards) == 36, "Not one-hot encoded anymore"
            return valid_remaining_cards

        def pop_random_valid_card(self, rule: GameRule, sampled_state: GameState):
            valid_cards = self._get_valid_cards(rule, sampled_state)
            card = np.random.choice(np.flatnonzero(valid_cards))
            assert self._remaining_cards[card] == 1, "Random valid card is not unexplored"
            assert self._played_cards[card] == 0, "Random valid card has already been played"
            self._remaining_cards[card] = 0
            self._played_cards[card] = 1

            return card

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
        self._rule = rule if rule else RuleSchieber()
        self._tree_policy = (
            tree_policy
            if tree_policy
            else lambda node, sample, p: self.UCB1_selection(node, sample, p, c=ucb1_c_param)
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

    def UCB1_selection(self, node: Node, sampled_state: GameState, player: int, c=1.0) -> Node:
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
        last_card = (
            state.get_card_played(state.nr_played_cards - 1)
            if state.nr_played_cards > 0
            else -1
        )

        last_player = -1
        if state.player >= 0:
            if state.nr_cards_in_trick > 0:
                # currently in a trick, player order is according to next_player
                # if upper has played, lower has not and [x] is current (going right to left)
                # a [b] C d  ->  previous was just the inverse of next_player
                last_player = next_player.index(state.player)
            elif state.nr_tricks > 0:
                # trick was just won, player is now the trick winner.
                # last player was the last player of the previous trick so 4th player after
                # the previous trick starter which is the player before the last trick starter because of wrap around.
                # 2: a [b] c d  ->  winner was b but last player was D, which is inverse of next_player for prev starter
                # 1: A B {C} D -> C started, went C-B-A-D, and winner was b
                last_player = next_player.index(state.trick_first_player[state.nr_tricks-1])

        root = self.Node(
            state,
            last_played_card=last_card,
            last_player=last_player,
            root_player=state.player,
            parent=None,
        )

        return self.start_mcts(root, timebudget)

    def start_mcts(self, node: Node, time_budget: float):
        """Does MCTS during a certain time_budget (in seconds) from node and returns the best card to play."""
        time_end = time.time() + time_budget

        while time.time() < time_end:
            self.mcts(node)

        # all the children must be valid here, since the root node has perfect information about move validity
        return max(node.children, key=lambda n: n.N).last_played_card

    def _sample_state(self, node: Node) -> GameState:
        remaining_cards = np.array(_get_remaining_cards_in_play_from_obs(node.known_state))
        assert (
            len(remaining_cards)
            == 36 - num_cards_in_hand(node.known_state.hand) - node.known_state.nr_played_cards
        ), "Remaining cards don't match known state"
        assert node.known_state.player == node.root_player, "It's not root player's turn"

        np.random.shuffle(remaining_cards)
        distributed_hands = np.array_split(remaining_cards, 3)
        num_cards_in_root_hand = num_cards_in_hand(node.known_state.hand)
        most_card_in_non_root_hand = len(distributed_hands[0])
        n_trick = node.known_state.nr_cards_in_trick

        # schema: ROOT MAX SECOND THIRD | max, second, third are descending
        if num_cards_in_root_hand == most_card_in_non_root_hand - 1:
            # one less in root's hand than the max of the others
            # could be 8999, 8998, 8988 so  1 <= n_trick <= 3
            assert 1 <= n_trick <= 3, "Root has 1 less card than others but <1 or >3 in trick"
        elif num_cards_in_root_hand == most_card_in_non_root_hand:
            # equal to max
            # could be 9999, 9998, 9988 so  0 <= n_trick <= 2
            assert 0 <= n_trick <= 2, "Root has same nr cards as others but >2 in trick"
        elif num_cards_in_root_hand == most_card_in_non_root_hand + 1:
            # one more than max
            # could be 9888, so n_trick == 3
            assert n_trick == 3, "Root has 1 card more than others but not 3 in trick"
        else:
            # difference > 1 -> cannot happen
            # things like 7988 are illegal, root played one too many and some other player one too few
            assert False, "More than 1 difference in hand sizes"

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
            # and for those who haven't (i >= nr_cards_in_trick), it should be 1 more.
            # if there are 0 in the trick or only the root player has played,
            # the number will be divisible by 3 and everyone should have floor(remaining / 3)
            # without adding anything to it.
            assert len(hand_indices) == (len(remaining_cards)) // 3 + (
                0
                if i < node.known_state.nr_cards_in_trick or len(remaining_cards) % 3 == 0
                else 1
            )

            # should not derive i from p due to order, just use next_player
            p = next_player[p]

        return state_from_observation(node.known_state, hands)

    def _selection(self, node: Node, sampled_state: GameState) -> Tuple[Node, GameState]:
        assert (
            node.last_sampled_state is None
        ), "Last sample was not reset correctly before entering selection"
        node.last_sampled_state = sampled_state

        next_node = node
        game_sim = None

        assert (
            next_node.known_state.nr_played_cards == sampled_state.nr_played_cards
        ), "Node & sample state out of sync"

        while not next_node.is_leaf and next_node.fully_expanded_for(
            self._rule, sampled_state
        ):
            if not game_sim:  # initialize GameSim only if needed
                game_sim = GameSim(self._rule)
                game_sim.init_from_state(sampled_state)

            next_node = self._tree_policy(next_node, sampled_state, node.known_state.player)

            # must advance sample state to keep up with newly chosen node
            # the tree policy can only select children that are valid
            # for the current state sample so this should always be a legal move.
            game_sim.action_play_card(next_node.last_played_card)
            sampled_state = game_sim.state

            assert (
                next_node.last_sampled_state is None
            ), "Last sample was not reset correctly before entering selection"

            next_node.last_sampled_state = sampled_state

        assert next_node.is_leaf or not next_node.fully_expanded_for(
            self._rule, sampled_state
        ), "selected node is not leaf or fully_expanded"
        assert (
            next_node.known_state.nr_played_cards == sampled_state.nr_played_cards
        ), "Node & sample state out of sync"

        return next_node, sampled_state

    def _expansion(self, node: Node, sampled_state: GameState) -> Tuple[Node, GameState]:
        assert not node.fully_expanded_for(
            self._rule, sampled_state
        ), "Fully expanded node in _expansion"

        assert node.last_sampled_state == sampled_state, "Last sampled state not in sync"

        # if selected node is terminal, take the payoff directly (rollout will end immediately)
        if node.is_terminal:
            return node, sampled_state

        # if selected has never been sampled before, do a rollout from it
        if not node.has_been_sampled:
            return node, sampled_state

        # if selected has already been sampled (but isn't fully expanded), select
        # random valid move and add a branch from this node. Then do a rollout from there.
        sampled_card = node.pop_random_valid_card(self._rule, sampled_state)

        game_sim = GameSim(self._rule)
        game_sim.init_from_state(sampled_state)
        game_sim.action_play_card(sampled_card)

        sampled_state = game_sim.state  # update sample state with sampled action

        next_node = ISMCTS.Node(
            observation_from_state(
                # removes the information we can't know, like
                # the hands of the others. However, it still
                # leaks information because it keeps who won
                # the last trick -> order of players.
                # may not be an issue because incompatible ones are filtered anyway.
                sampled_state,
                node.root_player,
            ),
            sampled_card,
            node.known_state.player,
            node.root_player,  # root player stays
            node,
        )
        next_node.last_sampled_state = sampled_state
        if node.children is None:
            node.children = []
        node.children.append(next_node)

        assert (
            next_node.known_state.nr_played_cards == node.known_state.nr_played_cards + 1
        ), "Child does not have 1 more card played than parent"
        assert (
            next_node.last_sampled_state.nr_played_cards
            == node.last_sampled_state.nr_played_cards + 1
        ), "Child does not have 1 more card played than parent"

        return next_node, sampled_state

    def _backpropagation(self, node: Node, payoffs: Payoffs):
        # Note: Only function allowed to read and reset the last_sampled_state field
        parent = node
        while parent is not None:
            parent.N += 1
            parent.W += payoffs

            assert parent.last_sampled_state is not None, "No last sample for node"
            if parent.children:
                for child in parent.get_children_consistent_with_sample(
                    parent.last_sampled_state
                ):
                    # children need to know how many times their parents were simulated
                    # and them being compatible. Room for performance improvements, surely.
                    # The fact that we need to iterate over get_children_consistent_with_sample
                    # leads to the introduction of last_sampled_state, which could be cut completely
                    # if we find a better time and place to count how many times this sample was available
                    # when the parent was sampled. Maybe in the iterator itself but that's not very clean.
                    child.parentN += 1
                    assert (
                        child.last_sampled_state is None
                    ), "Child still has last_sampled_state"

            parent.last_sampled_state = None
            parent = parent.parent

    def mcts(self, node: Node):
        assert not node.is_terminal, "Started mcts from terminal node"

        sampled_state = self._sample_state(node)
        self._assert_sample_validity(node, sampled_state)

        next_node, sampled_state = self._selection(node, sampled_state)
        next_node, sampled_state = self._expansion(next_node, sampled_state)
        payoffs = self._rollout(next_node, sampled_state)
        # backprop uses the last_sampled_state property, no need to pass sampled state.
        # could use last_sampled_state also for the other methods theoretically but
        # if possible I'd like to remove it somehow so this way it'll be easier to adapt.
        self._backpropagation(next_node, payoffs)

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
        assert np.array_equal(node.known_state.hand, sampled_state.hands[node.root_player])
        assert node.known_state == observation_from_state(sampled_state, node.root_player)
