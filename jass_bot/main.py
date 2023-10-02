import copy
import logging
import sys
from typing import Callable

from agents.alphabeta_agent import AlphaBetaAgent
from agents.game_tree_container import GameTreeContainer
from agents.minimax_agent import MiniMaxAgent
from agents.rule_based import RuleBasedAgent
from jass.agents.agent import Agent
from jass.agents.agent_cheating import AgentCheating
from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_util import *
from jass.game.const import *


def tournament_ABAB(agent_type: type | Callable[[], Agent | AgentCheating],
                    opponent_type: type | Callable[[], Agent | AgentCheating],
                    n_games=1000):
    logging.basicConfig(level=logging.INFO)

    agent = agent_type()
    opponent = opponent_type()

    arena = Arena(nr_games_to_play=n_games, save_filename='arena_games', cheating_mode=isinstance(agent, AgentCheating))
    arena.set_players(
        agent,
        opponent,
        agent_type(),
        opponent_type()
    )

    print(f'Playing {arena.nr_games_to_play} games')
    arena.play_all_games()
    print(f'Average Points Team 0 (ours): {arena.points_team_0.mean():.2f})')
    print(f'Average Points Team 1 (base): {arena.points_team_1.mean():.2f})')


def test_case_valid_card():
    schieber, trump, move_nr = RuleSchieber(), D, 3
    trick = np.array([C6, DJ, D10])
    # highest trump in this trick is DJ
    # lowest trump in this trick is D10
    # trumps lower than D10 = D8, D7, D6
    # trumps higher than DJ = None
    # indices are in order of DA, DK, DQ, DJ, D10, D9, D8, D7, D6
    print([DA, DK, DQ, DJ, D10, D9, D8, D7, D6])
    # therefore D10 > DJ meaning that in get_valid_cards the check lowest_trump_played < current_trick[2]
    # will think that D10 is higher than DJ and update lowest_trump_played leading to the invalid response
    # that only trumps below D10 are invalid even though all trumps should be invalid because DJ is the max.
    hand = get_cards_encoded([DA, D9, D6, S10])
    # player has something else than trump in hand and is therefore not allowed to play any lower trumps
    valid_cards = schieber.get_valid_cards(hand, trick, move_nr, trump)
    print(valid_cards)
    print(convert_one_hot_encoded_cards_to_str_encoded_list(valid_cards))
    assert valid_cards[DA] == 0  # lower than DJ, cannot play
    assert valid_cards[D9] == 0  # lower than DJ, cannot play
    assert valid_cards[D7] == 0  # lower than DJ, cannot play
    assert valid_cards[S10] == 1 # not trump, can play


if __name__ == "__main__":
    # tournament_ABAB(RuleBasedAgent, AgentRandomSchieber)
    tree_container = GameTreeContainer()
    # tournament_ABAB(lambda: MiniMaxAgent(tree_container, depth=2), AgentCheatingRandomSchieber, n_games=10)
    # tournament_ABAB(lambda: AlphaBetaAgent(tree_container, depth=2), AgentCheatingRandomSchieber, n_games=10)
    tournament_ABAB(lambda: AlphaBetaAgent(tree_container, depth=1), lambda: MiniMaxAgent(tree_container, depth=1), n_games=100)
    # test_case_valid_card()
    # print(count_colors(get_cards_encoded([DA, DQ, D6, S10, S7, C9])))
