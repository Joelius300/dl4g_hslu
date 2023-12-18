import copy
import logging
import math
import sys
from typing import Callable

from jass_bot.agents.MultiProcessingISMCTS import MultiProcessingISMCTS
from jass_bot.agent_definitions import TrumpDefs, CardDefs
from jass_bot.agent_definition import AgentDefinition
from jass.agents.agent_by_network import AgentByNetwork
from jass_bot.agents.CompositeAgent import CompositeAgent
from jass_bot.heuristics import graf
from jass_bot.agents.CheatingMCTS import CheatingMCTS
from jass_bot.agents.ISMCTS import ISMCTS
from jass_bot.agents.alphabeta_agent import AlphaBetaAgent
from jass_bot.agents.game_tree_container import GameTreeContainer
from jass_bot.agents.minimax_agent import MiniMaxAgent
from jass_bot.agents.rule_based import RuleBasedAgent
from jass.agents.agent import Agent
from jass.agents.agent_cheating import AgentCheating
from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_util import *
from jass.game.const import *
from jass_bot.strategies.card_strategy import CardStrategy
from jass_bot.strategies.trump_strategy import TrumpStrategy
from jass_bot.agents.ISMCTS import points_div_by_max, binary_payoff
from jass_bot.tournament import tournament_ABAB, round_robin, round_robin_sets


def compare_trump_strategies(time_budget=0.05, n_sets=100, **kwargs):
    return round_robin_sets(
        {
            "ISMCTS w/ Graf": AgentDefinition(
                TrumpDefs.graf(), CardDefs.ISMCTS(time_budget), True
            ),
            "ISMCTS w/ Random": AgentDefinition(
                TrumpDefs.random(), CardDefs.ISMCTS(time_budget), True
            ),
            "Random w/ Graf": AgentDefinition(TrumpDefs.graf(), CardDefs.random(), False),
            "Random w/ Random": AgentDefinition(TrumpDefs.random(), CardDefs.random(), False),
        },
        n_sets=n_sets,
        **kwargs,
    )


def check_failing():
    return round_robin_sets(
        {
            "Random": AgentDefinition(
                TrumpDefs.random(),
                CardDefs.random(),
            ),
            "Failing Trump": AgentDefinition(
                TrumpDefs.just_fail(),
                CardDefs.random(),
            ),
            "Failing Card": AgentDefinition(
                TrumpDefs.random(),
                CardDefs.just_fail(),
            ),
        },
        n_sets=2,
    )


def compare_payoff_functions(time_budget=0.05, n_games=100):
    # Evaluates different payoff functions to see which ones
    # perform better. Only works for versions of ISMCTS where
    # the payoff function is a parameter.
    # Found that binary-winner outperforms point-normalized consistently.
    payoff_functions = {
        "points-normalized": points_div_by_max,
        "binary-winner": binary_payoff,
    }

    return round_robin(
        {
            name: lambda: ISMCTS(time_budget=time_budget, get_payoffs=payoff_func)
            for name, payoff_func in payoff_functions.items()
        },
        n_games=n_games,
    )


def compare_normal_to_multiprocessing_mcts(
    time_budget=1.0, num_workers=4, point_threshold=1000
):
    arena = Arena(print_every_x_games=1, print_timings=True)
    a = MultiProcessingISMCTS(
        time_budget, num_workers=num_workers, ignore_same_player_safety=True
    )
    b = ISMCTS(time_budget, ignore_same_player_safety=True)
    arena.set_players(a, b, a, b)
    winner = arena.play_until_point_threshold(point_threshold)
    logging.info(f"Multiprocessing did {('NOT ' if winner == 1 else '')}outperform normal ISMCTS.")


def compare_different_c_param_values(time_budget: float, n_sets=4, **kwargs):
    sq2 = math.sqrt(2)
    # cs = [1, sq2, sq2 / 2] + [sq2 * 1.5 * i for i in range(1, 10, 2)]
    cs = [sq2, 3, 4, 5, 6, 7, 8]
    players = {
        f"c={c}": AgentDefinition(
            TrumpDefs.graf(),
            CardDefs.ISMCTS(time_budget, ucb1_c_param=c, ignore_same_player_safety=True),
        )
        for c in cs
    }

    return round_robin_sets(players, n_sets=n_sets, **kwargs)


def compare_different_c_param_values_long_multi(
    time_budget: float, n_games=-1, point_threshold=-1, num_workers=8
):
    cs = [7, 8.5, 10]
    players = {
        f"c={c}": MultiProcessingISMCTS(
            time_budget,
            ucb1_c_param=c,
            ignore_same_player_safety=True,
            num_workers=num_workers,
        )
        for c in cs
    }

    return round_robin(players, n_games, point_threshold)


def compare_baseline_against_remote(url: str, time_budget=0.5, point_threshold=1000):
    arena = Arena(print_every_x_games=1, print_timings=True)
    a = AgentByNetwork(url)
    b = ISMCTS(time_budget, ignore_same_player_safety=True)
    arena.set_players(a, b, a, b)
    winner = arena.play_until_point_threshold(point_threshold)
    logging.info(f"Remote {url} did {('NOT ' if winner == 1 else '')}outperform baseline ISMCTS.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("log.txt"), logging.StreamHandler()],
    )

    # tournament_ABAB(
    #     AgentByNetwork("http://localhost:8888/ISMCTS"), AgentRandomSchieber, n_games=3
    # )

    # compare_payoff_functions(time_budget=0.01)
    # compare_trump_strategies(0.05, n_sets=10)
    # check_failing()

    # time_budget = .05
    # arena = Arena(print_every_x_games=1)
    # a = lambda: CompositeAgent(
    #     TrumpStrategy.from_function(graf.graf_trump_selection),
    #     CardStrategy.from_agent(ISMCTS(time_budget)),
    # )
    # b = lambda: CompositeAgent(
    #     TrumpStrategy.from_agent(AgentRandomSchieber()),
    #     CardStrategy.from_agent(ISMCTS(time_budget)),
    # )
    # arena.set_players(a(), b(), a(), b())
    # arena.play_until_point_threshold(1000)

    # compare_normal_to_multiprocessing_mcts(1, 12, point_threshold=10000)
    # print(compare_different_c_param_values(0.2, n_sets=15))
    # print(compare_different_c_param_values_long_multi(9.5, point_threshold=500))
    # print(compare_trump_strategies(0.01, 2, num_workers=0))

