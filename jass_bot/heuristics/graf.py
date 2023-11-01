"""
Jass heuristic values from Daniel Graf from 2009: "Jassen auf Basis der Spieltheorie"
"""

from functools import cache

import numpy as np

from jass.game.const import UNE_UFE, OBE_ABE, MAX_TRUMP, PUSH
from jass.game.game_observation import GameObservation

# score if the color is trump
trump_scores = [15, 10, 7, 25, 6, 19, 5, 5, 5]
# score if the color is not trump
no_trump_scores = [9, 7, 5, 2, 1, 0, 0, 0, 0]
# score if obenabe is selected (all colors)
obenabe_scores = [14, 10, 8, 7, 5, 0, 5, 0, 0]
# score if uneufe is selected (all colors)
uneufe_scores = [0, 2, 1, 1, 5, 5, 7, 9, 11]
# score threshold to exceed, otherwise push
push_threshold = 68


@cache
def get_graf_scores(trump: int):
    """
    Get the (36,) ndarray that contains the graf scores according to the current trump.
    """
    if trump == OBE_ABE:
        scores = obenabe_scores * 4
    elif trump == UNE_UFE:
        scores = uneufe_scores * 4
    else:
        scores = (
            no_trump_scores * trump
            + trump_scores
            + no_trump_scores * (3 - trump)
        )

    return np.array(scores)


def graf_trump_selection(obs: GameObservation) -> int:
    def points_for_trump(trump: int):
        return np.sum(get_graf_scores(trump) * obs.hand)

    scores_for_trumps = [points_for_trump(i) for i in range(MAX_TRUMP + 1)]
    best_trump = np.argmax(scores_for_trumps)

    if scores_for_trumps[best_trump] < push_threshold and obs.forehand == -1:
        return PUSH

    return int(best_trump)
