import copy
import logging
import sys
from typing import Callable

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
from jass_bot.tournament import tournament_ABAB, round_robin


def compare_trump_strategies(timebudget=0.05, n_games=100):
    random_agent = AgentRandomSchieber()
    return round_robin(
        {
            "ISMCTS w/ Graf": lambda: CompositeAgent(
                TrumpStrategy.from_function(graf.graf_trump_selection),
                CardStrategy.from_agent(ISMCTS(timebudget)),
            ),
            "ISMCTS w/ Random": lambda: CompositeAgent(
                TrumpStrategy.from_agent(random_agent),
                CardStrategy.from_agent(ISMCTS(timebudget)),
            ),
            "Random w/ Graf": lambda: CompositeAgent(
                TrumpStrategy.from_function(graf.graf_trump_selection),
                CardStrategy.from_agent(random_agent),
            ),
            "Random w/ Random": AgentRandomSchieber,
        },
        n_games=n_games,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tournament_ABAB(
        AgentByNetwork("http://localhost:8888/ISMCTS"), AgentRandomSchieber, n_games=5
    )
