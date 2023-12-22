from collections import namedtuple

from jass_bot.agent_definitions import get_trump_strat, get_card_strat
from jass_bot.agents.composite_agent import CompositeAgent
from jass_bot.agents.multi_player_agent_container import MultiPlayerAgentContainer
from jass.agents.agent import Agent

AgentDefinition = namedtuple(
    "AgentDefinition", ["trump", "card", "needs_user_wrapper", "random_fallback"],
    defaults=[False, True],  # defaults apply from left to right but right-aligned
)


def create_agent(definition: AgentDefinition) -> Agent:
    trump, card, needs_wrapper, random_fallback = definition
    MAX_AGENTS_NEEDED = 4  # an agent that needs to handle all 4 players
    agent_generator = (
        CompositeAgent(get_trump_strat(trump, random_fallback), get_card_strat(card, random_fallback))
        for _ in range(MAX_AGENTS_NEEDED)
    )

    if needs_wrapper:
        agent = MultiPlayerAgentContainer.from_agents(agent_generator)
    else:
        agent = next(agent_generator)

    return agent
