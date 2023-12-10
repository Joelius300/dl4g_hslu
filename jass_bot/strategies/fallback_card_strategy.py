from logging import Logger

from jass.game.game_observation import GameObservation
from strategies.card_strategy import CardStrategy


class FallbackCardStrategy(CardStrategy):
    """A card strategy that falls back to another one in case of error."""

    def __init__(self, strategy: CardStrategy, fallback: CardStrategy, fallback_name=None):
        self.logger = Logger(__name__)
        self.strategy = strategy
        self.fallback = fallback
        self.fallback_name = fallback_name

    def action_play_card(self, obs: GameObservation) -> int:
        try:
            return self.strategy.action_play_card(obs)
        except Exception as e:
            self.logger.warning(
                "Card strategy failed, using fallback strategy"
                f"{(f' ({self.fallback_name})' if self.fallback_name else '')}!",
                exc_info=e,
            )
            return self.fallback.action_play_card(obs)
