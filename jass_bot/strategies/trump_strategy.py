from abc import ABC, abstractmethod
from typing import Callable

from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation


class TrumpStrategy(ABC):
    @abstractmethod
    def action_trump(self, obs: GameObservation) -> int:
        pass

    @staticmethod
    def from_function(func: Callable[[GameObservation], int]):
        """Returns a trump strategy that delegates calls a function."""
        return FunctionTrumpStrategy(func)

    @staticmethod
    def from_agent(agent: Agent):
        """Returns a trump strategy that delegates calls to an agent instance."""
        return FunctionTrumpStrategy(agent.action_trump)


class FunctionTrumpStrategy(TrumpStrategy):
    def __init__(self, func: Callable[[GameObservation], int]):
        self.func = func

    def action_trump(self, obs: GameObservation) -> int:
        return self.func(obs)
