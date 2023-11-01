import itertools
from typing import Callable

from jass.agents.agent import Agent
from jass.agents.agent_cheating import AgentCheating
from jass.arena.arena import Arena


def tournament_ABAB(
    ours: Agent | AgentCheating | type | Callable[[], Agent | AgentCheating],
    base: Agent | AgentCheating | type | Callable[[], Agent | AgentCheating],
    n_games=1000, save_filename="arena_games", print_every=5,
):
    """
    Run an ABAB tournament with two types of agents: ours (A) and base (B).
    If given an instance, will use that instance for all players.
    If given factory method or type, will create new instance per player.
    """
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
        nr_games_to_play=n_games,
        save_filename=save_filename,
        print_every_x_games=print_every,
        cheating_mode=isinstance(ours, AgentCheating),
    )
    arena.set_players(ours, base, ours_2, base_2)

    print(f"Playing {arena.nr_games_to_play} games")
    arena.play_all_games()
    mean_ours: float = arena.points_team_0.mean()
    mean_base: float = arena.points_team_1.mean()
    std_ours: float = arena.points_team_0.std()
    std_base: float = arena.points_team_1.std()
    print(f"Avg Points Team 0 (ours): {mean_ours:.2f} with std {std_ours:.2f}")
    print(f"Avg Points Team 1 (base): {mean_base:.2f} with std {std_base:.2f}")

    return mean_ours, mean_base, std_ours, std_base, arena.points_team_0, arena.points_team_1


def round_robin(players: dict[str, Agent | AgentCheating | type | Callable[[], Agent | AgentCheating]], n_games: int, **kwargs):
    scores = {}
    matchups = {}
    for [a, b] in itertools.combinations(players.keys(), 2):
        print(f"{a} vs. {b}")
        mean_a, mean_b, std, *_ = tournament_ABAB(players[a], players[b], n_games=n_games, **kwargs)
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
    for opponent, (score_best, score_opp, std) in sorted(matchups[best_player].items(), key=lambda m: m[1][0]):
        print(f"  vs. {opponent}: {score_best:.2f} to {score_opp:.2f} (std {std:.2f})")

    return scores, matchups
