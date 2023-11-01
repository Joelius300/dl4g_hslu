from abc import ABC, abstractmethod
from typing import Callable

from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation


class CardStrategy(ABC):
    @abstractmethod
    def action_play_card(self, obs: GameObservation) -> int:
        pass

    @staticmethod
    def from_function(func: Callable[[GameObservation], int]):
        """Returns a card strategy that delegates calls a function."""
        return FunctionCardStrategy(func)

    @staticmethod
    def from_agent(agent: Agent):
        """Returns a card strategy that delegates calls to an agent instance."""
        return FunctionCardStrategy(agent.action_play_card)


class FunctionCardStrategy(CardStrategy):
    def __init__(self, func: Callable[[GameObservation], int]):
        self.func = func

    def action_play_card(self, obs: GameObservation) -> int:
        return self.func(obs)
