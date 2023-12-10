from logging import Logger

from jass.game.game_observation import GameObservation
from strategies.trump_strategy import TrumpStrategy


class FallbackTrumpStrategy(TrumpStrategy):
    """A trump strategy that falls back to another one in case of error."""

    def __init__(self, strategy: TrumpStrategy, fallback: TrumpStrategy, fallback_name=None):
        self.logger = Logger(__name__)
        self.strategy = strategy
        self.fallback = fallback
        self.fallback_name = fallback_name

    def action_trump(self, obs: GameObservation) -> int:
        try:
            return self.strategy.action_trump(obs)
        except Exception as e:
            self.logger.warning(
                "Trump strategy failed, using fallback strategy"
                f"{(f' ({self.fallback_name})' if self.fallback_name else '')}!",
                exc_info=e,
            )
            return self.fallback.action_trump(obs)
