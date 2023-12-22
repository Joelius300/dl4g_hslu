from typing import Callable, Iterable

from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation


class MultiPlayerAgentContainer(Agent):
    """
    A container to allow a single agent instance to handle multiple players, even when a single instance
    of the contained agent is dependent on the player not changing (even if just for sanity checks).
    Allows clear separation of players.
    """
    def __init__(self, agent_type_or_factory: type | Callable[[], Agent]):
        self.agent_factory = agent_type_or_factory
        self.agents = {}

    def action_trump(self, obs: GameObservation) -> int:
        return self.get_player_agent(obs.player).action_trump(obs)

    def action_play_card(self, obs: GameObservation) -> int:
        return self.get_player_agent(obs.player).action_play_card(obs)

    def get_player_agent(self, player: int) -> Agent:
        if player not in self.agents:
            agent = self.agent_factory()
            assert agent is not None and isinstance(agent, Agent), "agent factory returned no or invalid agent"
            self.agents[player] = agent

        return self.agents[player]

    @classmethod
    def from_agents(cls, agents: Iterable[Agent]):
        iterator = iter(agents)
        # will keep on taking from the agents iterable until consumed.
        return MultiPlayerAgentContainer(iterator.__next__)
