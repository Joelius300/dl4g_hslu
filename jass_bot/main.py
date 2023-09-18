import logging
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber

from jass_bot.agents.rule_based import RuleBasedAgent

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    agent_type = RuleBasedAgent
    opponent_type = AgentRandomSchieber

    arena = Arena(nr_games_to_play=1000, save_filename='arena_games')
    arena.set_players(
        agent_type(),
        opponent_type(),
        agent_type(),
        opponent_type()
    )

    print(f'Playing {arena.nr_games_to_play} games')
    arena.play_all_games()
    print(f'Average Points Team 0: {arena.points_team_0.mean():.2f})')
    print(f'Average Points Team 1: {arena.points_team_1.mean():.2f})')
