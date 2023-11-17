import concurrent.futures
import itertools
import os
from typing import Callable, Optional

import numpy as np

from jass.agents.agent import Agent
from jass.agents.agent_cheating import AgentCheating
from jass.arena.arena import Arena


def tournament_ABAB(
    ours: Agent | AgentCheating | type | Callable[[], Agent | AgentCheating],
    base: Agent | AgentCheating | type | Callable[[], Agent | AgentCheating],
    n_games=-1,
    point_threshold=-1,
    save_filename: Optional[str] = "arena_games",
    print_every=5,
):
    """
    Run an ABAB tournament with two types of agents: ours (A) and base (B).
    If given an instance, will use that instance for all players.
    If given factory method or type, will create new instance per player.
    """
    assert (n_games > 0) ^ (
        point_threshold > 0
    ), "One of n_games or point_threshold must be positive"

    if not isinstance(ours, Agent) and not isinstance(ours, AgentCheating):
        ours_2 = ours()
        ours = ours()
    else:
        # reuse same instance
        ours_2 = ours

    if not isinstance(base, Agent) and not isinstance(base, AgentCheating):
        base_2 = base()
        base = base()
    else:
        # use same instance
        base_2 = base

    arena = Arena(
        save_filename=save_filename,
        print_every_x_games=print_every,
        cheating_mode=isinstance(ours, AgentCheating),
    )
    arena.set_players(ours, base, ours_2, base_2)

    if n_games > 0:
        print(f"Playing {n_games} games")
        winner = arena.play_games(n_games)
    else:
        print(f"Playing to {point_threshold} point")
        winner = arena.play_until_point_threshold(point_threshold)

    mean_ours: float = arena.points_team_0.mean()
    mean_base: float = arena.points_team_1.mean()
    std_ours: float = arena.points_team_0.std()
    std_base: float = arena.points_team_1.std()
    games_played = arena.nr_games_played
    print(f"Winner: {winner} ({('ours' if winner == 0 else 'base')}) after {games_played} games")
    print(f"Avg Points Team 0 (ours): {mean_ours:.2f} with std {std_ours:.2f}")
    print(f"Avg Points Team 1 (base): {mean_base:.2f} with std {std_base:.2f}")

    return winner, mean_ours, mean_base, std_ours, std_base, games_played, arena.points_team_0, arena.points_team_1


def tournament_multiple_sets(
    ours: type | Callable[[], Agent | AgentCheating],
    base: type | Callable[[], Agent | AgentCheating],
    n_sets: int,
    point_threshold=1000,
    num_workers=None,
):
    num_workers = num_workers if num_workers is not None else os.cpu_count()

    def run_tournament(_i):
        winner, mean_ours, mean_base, games_played, *_ = tournament_ABAB(ours, base, point_threshold=point_threshold, save_filename=None)
        return winner, mean_ours, mean_base, games_played

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_tournament, range(n_sets)))
        results = np.array(results)
        wins_1 = np.count_nonzero(results[:, 0])
        wins_0 = results.shape[0] - wins_1
        n_games = results[:, 3]
        total_games = np.sum(n_games)
        avg_games_played = np.mean(n_games)
        means = (results[:, [1, 2]] * n_games / total_games).ravel()

        return wins_0, wins_1, means[0], means[1], avg_games_played, total_games


def round_robin_games(
    players: dict[str, Agent | AgentCheating | type | Callable[[], Agent | AgentCheating]],
    n_games: int,
    **kwargs,
):
    scores = {}
    matchups = {}
    for [a, b] in itertools.combinations(players.keys(), 2):
        print(f"{a} vs. {b}")
        _, mean_a, mean_b, std, *_ = tournament_ABAB(
            players[a], players[b], n_games=n_games, **kwargs
        )
        print()
        if a not in matchups:
            matchups[a] = {}
        if b not in matchups:
            matchups[b] = {}

        matchups[a][b] = (mean_a, mean_b, std)
        matchups[b][a] = (mean_b, mean_a, std)

        if a not in scores:
            scores[a] = []
        if b not in scores:
            scores[b] = []

        scores[a].append(mean_a - mean_b)
        scores[b].append(mean_b - mean_a)

    best_player = max(scores, key=lambda p: sum(scores[p]))

    print(f"Best player is {best_player} who scored as follows:")
    # order by our score ascending which should roughly give an ordering of the next best players ascending
    # (it's more intuitive to see the first listed opponent and think that's best of them, instead of the easiest)
    for opponent, (score_best, score_opp, std) in sorted(
        matchups[best_player].items(), key=lambda m: m[1][0]
    ):
        print(f"  vs. {opponent}: {score_best:.2f} to {score_opp:.2f} (std {std:.2f})")

    return scores, matchups


def round_robin_sets(
        players: dict[str, type | Callable[[], Agent | AgentCheating]],
        n_sets: int,
        **kwargs,
):
    scores = {}
    matchups = {}
    for [a, b] in itertools.combinations(players.keys(), 2):
        print(f"{a} vs. {b}")
        wins_a, wins_b, mean_a, mean_b, games_avg, games_total = tournament_multiple_sets(
            players[a], players[b], n_sets=n_sets, **kwargs
        )
        print()
        if a not in matchups:
            matchups[a] = {}
        if b not in matchups:
            matchups[b] = {}

        matchups[a][b] = (wins_a, wins_b, mean_a, mean_b, games_avg, games_total)
        matchups[b][a] = (wins_a, wins_b, mean_b, mean_a, games_avg, games_total)

        if a not in scores:
            scores[a] = []
        if b not in scores:
            scores[b] = []

        total_sets = wins_b + wins_a
        scores[a].append(wins_a / total_sets)
        scores[b].append(wins_b / total_sets)

    best_player = max(scores, key=lambda p: sum(scores[p]))

    print(f"Best player is {best_player} who scored as follows:")
    # order by our wins ascending which should roughly give an ordering of the next best players ascending
    # (it's more intuitive to see the first listed opponent and think that's best of them, instead of the easiest)
    for opponent, (wins_best, wins_opp, score_best, score_opp, games_avg, games_total) in sorted(
            matchups[best_player].items(), key=lambda m: m[1][0]
    ):
        win_rate = wins_best / (wins_best + wins_opp)
        print(f"  vs. {opponent}: {wins_best} to {wins_opp} wins ({win_rate:.2%}) "
              f"(scores {score_best:.2f} to {score_opp:.2f}) "
              f"with avg set length {games_avg} and total games played {games_total}")

    return scores, matchups
