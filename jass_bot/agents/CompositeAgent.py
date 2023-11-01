from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation
from strategies.card_strategy import CardStrategy
from strategies.trump_strategy import TrumpStrategy


class CompositeAgent(Agent):
    def __init__(self, trump_strategy: TrumpStrategy, card_strategy: CardStrategy):
        self.trump_strategy = trump_strategy
        self.card_strategy = card_strategy

    def action_trump(self, obs: GameObservation) -> int:
        return self.trump_strategy.action_trump(obs)

    def action_play_card(self, obs: GameObservation) -> int:
        return self.card_strategy.action_play_card(obs)
