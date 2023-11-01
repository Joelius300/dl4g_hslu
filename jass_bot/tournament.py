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
    print(f"Avg Points Team 0 (ours): {arena.points_team_0.mean():.2f} with std {arena.points_team_0.std():.2f}")
    print(f"Avg Points Team 1 (base): {arena.points_team_1.mean():.2f} with std {arena.points_team_1.std():.2f}")

    return arena.points_team_0, arena.points_team_1
