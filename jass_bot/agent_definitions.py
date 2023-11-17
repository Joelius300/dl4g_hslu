from collections import namedtuple

from agents.CompositeAgent import CompositeAgent
from agents.ISMCTS import ISMCTS
from agents.MultiPlayerAgentContainer import MultiPlayerAgentContainer
from heuristics import graf
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from strategies.card_strategy import CardStrategy
from strategies.trump_strategy import TrumpStrategy

AgentDefinition = namedtuple("AgentDefinition", ["trump", "card", "needs_user_wrapper"])


def create_agent(definition: AgentDefinition) -> Agent:
    trump, card, needs_wrapper = definition
    MAX_AGENTS_NEEDED = 4  # an agent that needs to handle all 4 players
    agent_generator = (CompositeAgent(get_trump_strat(trump), get_card_strat(card)) for _ in range(MAX_AGENTS_NEEDED))
    if needs_wrapper:
        agent = MultiPlayerAgentContainer.from_agents(agent_generator)
    else:
        agent = next(agent_generator)

    return agent


def get_trump_strat(trump_def: dict) -> TrumpStrategy:
    name = trump_def["name"]
    if name == "random":
        return TrumpStrategy.from_agent(AgentRandomSchieber())
    elif name == "graf":
        return TrumpStrategy.from_function(graf.graf_trump_selection)


def get_card_strat(card_def: dict) -> CardStrategy:
    name = card_def["name"]
    if name == "random":
        return CardStrategy.from_agent(AgentRandomSchieber())
    elif name == "ISMCTS":
        return CardStrategy.from_agent(ISMCTS(time_budget=card_def["time_budget"]))


class CardDefs:
    @classmethod
    def random(cls):
        return dict(name="random")

    @classmethod
    def ISMCTS(cls, time_budget: float):
        return dict(name="ISMCTS", time_budget=time_budget)


class TrumpDefs:
    @classmethod
    def random(cls):
        return dict(name="random")

    @classmethod
    def graf(cls):
        return dict(name="graf")

