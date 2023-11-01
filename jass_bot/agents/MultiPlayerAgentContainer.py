from typing import Callable

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
            self.agents[player] = self.agent_factory()

        return self.agents[player]
