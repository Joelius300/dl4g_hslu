import logging
import math
import os

from jass_bot.agents.MultiProcessingISMCTS import MultiProcessingISMCTS
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.service.player_service_app import PlayerServiceApp

DEFAULT_TIME_BUDGET = 5
DEFAULT_NUM_WORKERS = -1
DEFAULT_C_PARAM = math.sqrt(2)
DEFAULT_LOGGING_LEVEL = logging.WARNING
# name of the package that contains the code for the flask application
FLASK_PACKAGE_NAME = "jass_bot"


def create_app():
    logging_level = os.environ.get("LOGGING_LEVEL", DEFAULT_LOGGING_LEVEL)
    time_budget = float(os.environ.get("TIME_BUDGET", DEFAULT_TIME_BUDGET))
    num_workers = int(os.environ.get("NUM_WORKERS", DEFAULT_NUM_WORKERS))
    c_param = float(os.environ.get("C_PARAM", DEFAULT_C_PARAM))

    logging.basicConfig(level=logging_level)

    app = PlayerServiceApp(FLASK_PACKAGE_NAME)

    app.add_player("random", AgentRandomSchieber())
    # no need for fallback because flask service will handle that and return internal server error
    # and the Jass Server will then use a fallback player (random) to play for me.
    # Also, can ignore_same_player_safety as the implementation is now state-free and tested
    # to ensure there aren't any cross-player issues. This removes the need for MultiPlayerAgentContainer.
    players = os.environ["PLAYERS"].split(",")
    for player_name in players:
        player = MultiProcessingISMCTS(
            float(os.environ.get("TIME_BUDGET_" + player_name, time_budget)),
            num_workers=int(os.environ.get("NUM_WORKERS_" + player_name, num_workers)),
            ucb1_c_param=float(os.environ.get("C_PARAM_" + player_name, c_param)),
            ignore_same_player_safety=True,
        )
        app.add_player(player_name, player)
        logging.info(
            f"Player '{player_name}': time budget {player.time_budget}, num workers {player.num_workers}, "
            f"c param {player.ucb1_c_param}"
        )

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8888)
