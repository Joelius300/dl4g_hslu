import dvc.api
import dvclive.live

from agent_definitions import TrumpDefs, CardDefs
from agent_definition import AgentDefinition
from tournament import tournament_multiple_sets

STAGE_NAME = "evaluate_trump_model"

if __name__ == "__main__":
    params = dvc.api.params_show(STAGE_NAME)[STAGE_NAME]
    time_budget = params["time_budget"]
    n_sets = params["n_sets"]
    checkpoint_path = params["checkpoint_path"]
    point_threshold = params.get("point_threshold", 1000)
    num_workers = params.get("num_workers", None)
    ismcts = CardDefs.ISMCTS(time_budget)
    base = AgentDefinition(TrumpDefs.graf(), ismcts, True)

    ours = AgentDefinition(TrumpDefs.model(checkpoint_path), ismcts, True)
    (
        wins_ours,
        wins_base,
        means_ours,
        means_base,
        avg_games_played,
        total_games,
    ) = tournament_multiple_sets(ours, base, n_sets, point_threshold, num_workers)

    # this should log to the current experiment if there is one but not if there isn't. Not sure if this is correct.
    live = dvclive.Live(resume=True, save_dvc_exp=False)
    live.log_metric("win_rate", wins_ours / (wins_ours + wins_base))
    live.log_metric("mean_points", means_ours)
    live.log_metric("mean_points_graf", means_base)
    live.log_metric("avg_set_length", avg_games_played)
    live.log_metric("total_games", total_games)
