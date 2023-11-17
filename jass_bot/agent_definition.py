from collections import namedtuple

from agent_definitions import get_trump_strat, get_card_strat
from agents.CompositeAgent import CompositeAgent
from agents.MultiPlayerAgentContainer import MultiPlayerAgentContainer
from jass.agents.agent import Agent

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
