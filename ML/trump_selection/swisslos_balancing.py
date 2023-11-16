import dvc.api
import logging
import pandas as pd
from jass.game.const import card_strings
import numpy as np

STAGE_NAME = "swisslos_balancing"
PLAYER_ALL_STATS_PATH = "data/player_all_stat.json"
TRUMP_DATASET_PATH = "data/2018_10_18_trump.csv"
TRUMP_DATASET_BALANCED_PATH = "data/swisslos_balanced.csv"

logger = logging.getLogger(STAGE_NAME)


def get_top_players(top_p: float, n_games_threshold: int):
    player_stats = pd.read_json(PLAYER_ALL_STATS_PATH)
    rounded_stats = player_stats.round(0)
    sorted_above_threshold = rounded_stats.query(f"nr >= {n_games_threshold}").sort_values(
        ["mean", "std", "nr"], ascending=[False, True, False]
    )
    top_n = int(top_p * sorted_above_threshold.shape[0])

    return sorted_above_threshold.head(top_n)


def load_trump_dataset():
    features = np.append(card_strings, ["FH"])
    cols = np.append(features, ["user", "trump"])

    trumps = pd.read_csv(TRUMP_DATASET_PATH, header=None, names=cols)
    # for some reason, this dataset has 0 = push is an option, 1 = push is not an option.
    # but push should only be an option if FH=1 so invert it here.
    trumps["FH"] = 1 - trumps["FH"]

    return trumps


def get_trump_dataset_of_players(player_ids: pd.Series):
    dataset = load_trump_dataset()
    return dataset[dataset["user"].isin(player_ids)]


def get_balanced_dataset(dataset: pd.DataFrame, random_state=42):
    value_counts = dataset[["FH", "trump"]].value_counts()
    min_n = value_counts.min()
    selected = []
    for fh in [1, 0]:
        for trump in range(6 + fh):
            df = dataset.query(f"FH == {fh} & trump == {trump}").sample(
                min_n, random_state=random_state
            )
            selected.append(df)

    balanced = pd.concat(selected)

    return balanced, min_n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Note: I'm not 100% sure how to use this func, as it always seems to return only the params
    # that I specified in the dependencies to this stage, even when I don't pass a stage.
    params = dvc.api.params_show(stages=STAGE_NAME)[STAGE_NAME]
    top_p = params["top_p"]
    n_games_threshold = params["n_games_threshold"]
    random_state = params["random_state"]

    top_players = get_top_players(top_p, n_games_threshold)

    # weighted average of the mean scores of all players with the number of played games as weight
    mean_mean = ((top_players["mean"] * top_players["nr"]) / top_players["nr"].sum()).sum()
    contains_anonymous = 0 in top_players["id"]
    logger.info(
        f"Selected top {top_players.shape[0]} players with more than {n_games_threshold} games for a "
        f"player base with {top_players['nr'].sum()} games played combined and an average score of {mean_mean:.2f}. "
        f"The anonymous player base IS{('' if contains_anonymous else ' NOT')} in this selection."
    )

    dataset = get_trump_dataset_of_players(top_players["id"])
    balanced, samples_per_trump = get_balanced_dataset(dataset, random_state)

    logger.info(
        f"Balanced dataset from original size {dataset.shape[0]} down to {balanced.shape[0]} "
        f"with {samples_per_trump} samples per trump and forehand/rearhand."
    )

    balanced.to_csv(TRUMP_DATASET_BALANCED_PATH)
