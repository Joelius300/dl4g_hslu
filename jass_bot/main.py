import copy
import logging
import sys
from typing import Callable

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
from jass_bot.tournament import tournament_ABAB, round_robin_games, round_robin_sets


def compare_trump_strategies(time_budget=0.05, n_sets=100):
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

    return round_robin_games(
        {
            name: lambda: ISMCTS(time_budget=time_budget, get_payoffs=payoff_func)
            for name, payoff_func in payoff_functions.items()
        },
        n_games=n_games,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # tournament_ABAB(
    #     AgentByNetwork("http://localhost:8888/ISMCTS"), AgentRandomSchieber, n_games=3
    # )

    # compare_payoff_functions(time_budget=0.01)
    # compare_trump_strategies(0.05, n_sets=10)
    check_failing()

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
