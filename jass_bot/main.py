import copy
import logging
import sys
from typing import Callable

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # tournament_ABAB(RuleBasedAgent, AgentRandomSchieber)
    # tree_container = GameTreeContainer()
    # tournament_ABAB(lambda: MiniMaxAgent(tree_container, depth=1), AgentCheatingRandomSchieber, n_games=200)
    # tournament_ABAB(lambda: AlphaBetaAgent(tree_container, depth=2), AgentCheatingRandomSchieber, n_games=50)
    # tournament_ABAB(lambda: AlphaBetaAgent(tree_container, depth=3), lambda: MiniMaxAgent(tree_container, depth=1), n_games=30)
    # test_case_valid_card()
    # print(count_colors(get_cards_encoded([DA, DQ, D6, S10, S7, C9])))
    # tournament_ABAB(lambda: CheatingMCTS(timebudget=0.1), AgentCheatingRandomSchieber, n_games=100)

    # tournament_ABAB(lambda: ISMCTS(timebudget=.03), AgentRandomSchieber, n_games=100)
    # round_robin({
    #     "ISMCTS (tb=.01)": lambda: ISMCTS(timebudget=.01),
    #     "ISMCTS (tb=.1)": lambda: ISMCTS(timebudget=.1),
    #     "Random": AgentRandomSchieber
    # }, n_games=50)

    timebudget = .05
    random_agent = AgentRandomSchieber()
    round_robin({
        "ISMCTS w/ Graf": lambda: CompositeAgent(TrumpStrategy.from_function(graf.graf_trump_selection), CardStrategy.from_agent(ISMCTS(timebudget))),
        "ISMCTS w/ Random": lambda: CompositeAgent(TrumpStrategy.from_agent(random_agent), CardStrategy.from_agent(ISMCTS(timebudget))),
        "Random w/ Graf": lambda: CompositeAgent(TrumpStrategy.from_function(graf.graf_trump_selection), CardStrategy.from_agent(random_agent)),
        "Random w/ Random": AgentRandomSchieber,
    }, n_games=500)
