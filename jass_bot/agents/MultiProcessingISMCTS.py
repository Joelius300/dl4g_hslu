from __future__ import annotations

import logging
import math
import multiprocessing
import os
import random
import time
from itertools import repeat
from typing import Optional

import pandas as pd

from jass.game.game_rule import GameRule
from jass.game.game_util import *
from jass_bot.agents.ISMCTS import ISMCTS


class MultiProcessingISMCTS(ISMCTS):
    """
    Uses the same underlying methods as ISMCTS but utilizes root parallelization
    <https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf> to improve strength.

    Using multiprocessing only makes sense for a considerable time-budget to justify the overhead of multiple workers.
    """

    def __init__(
        self,
        time_budget: float,
        rule: GameRule = None,
        ucb1_c_param: Optional[float] = math.sqrt(2),
        num_workers=-1,
        ignore_same_player_safety=False,
    ):
        super().__init__(
            time_budget,
            rule,
            ucb1_c_param,
            ignore_same_player_safety,
        )
        self.num_workers = num_workers if num_workers >= 0 else (os.cpu_count() - 1)
        if time_budget < 0.1:
            self._logger.warn("Low time_budget for multiprocessing mcts, might perform badly.")

        self._logger.info(
            f"Initialized root parallelized ISMCTS with {self.num_workers} workers "
            f"and {time_budget}s time budget. UCB1 C param = {ucb1_c_param}."
        )

    def mcts_until_time(self, node, time_end, seed=-1):
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)

        # modifies the node in place!
        while time.time() < time_end:
            self.mcts(node)

        # all the children must be valid here, since the root node has perfect information about move validity.
        return {n.last_played_card: n.N for n in node.children} if node.children else None

    def start_mcts(self, node: ISMCTS.Node, time_budget: float):
        """Does MCTS during a certain time_budget (in seconds) from node and returns the best card to play."""
        # 25 milliseconds to account for overhead -> quite a lot actually
        time_to_do_the_rest = 0.025
        time_end = time.time() + time_budget - time_to_do_the_rest

        if self.num_workers == 0:
            # run on this thread
            total = self.mcts_until_time(node, time_end)
            results = [total]
        else:
            # start new processes and do multiprocessing
            with multiprocessing.Pool(self.num_workers) as executor:
                results = list(
                    executor.starmap(
                        # repeat((node, time_end), self.num_workers),
                        # This above does not work because then all the MCTS trees will run the same thing basically
                        # due to the same pseudo-randomness at rollouts. This will just be a less efficient version
                        # of normal ISMCTS. Instead, pass each worker their own, individual seed, to ensure different
                        # results for each tree and higher statistical variance in the outputs.
                        self.mcts_until_time,
                        [
                            (n, e, i)
                            for i, (n, e) in enumerate(
                                # node is pickled (cloned, effectively) so the same instance can be repeated
                                # without conflict. just start as many mcts runs as there are workers to occupy each.
                                repeat((node, time_end), self.num_workers)
                            )
                        ],
                    )
                )

        totals = {}
        # combine results from all the trees. Trees are not weighted, all are treated the same.
        not_populated = 0
        for total in results:
            if not total:
                not_populated += 1
                continue

            for card, n in total.items():
                if card in totals:
                    totals[card] += n
                else:
                    totals[card] = n

        if not_populated > 0:
            self._logger.warn(
                f"{not_populated} / {len(results)} roots were not populated (worker didn't have enough time?)"
            )

        if self._logger.getEffectiveLevel() <= logging.DEBUG:
            df = pd.DataFrame(results)
            df = df.rename(columns=lambda x: card_strings[x])
            self._logger.debug(df.to_string())

        best_card = max(totals, key=totals.get)
        self._logger.debug(
            f"Selected '{card_strings[best_card]}' with {totals[best_card]} total visits across {len(results)} trees."
        )

        return best_card
