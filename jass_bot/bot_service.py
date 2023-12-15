import logging
import math
import os

from agents.MultiProcessingISMCTS import MultiProcessingISMCTS
from jass_bot.agents.MultiPlayerAgentContainer import MultiPlayerAgentContainer
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.service.player_service_app import PlayerServiceApp
from jass_bot.agents.ISMCTS import ISMCTS

DEFAULT_TIME_BUDGET = 5
DEFAULT_NUM_WORKERS = -1
DEFAULT_C_PARAM = math.sqrt(2)
DEFAULT_LOGGING_LEVEL = logging.WARNING
# name of the package that contains the code for the flask application
FLASK_PACKAGE_NAME = "jass_bot"


def create_app():
    time_budget = float(os.environ.get("TIME_BUDGET", DEFAULT_TIME_BUDGET))
    num_workers = int(os.environ.get("NUM_WORKERS", DEFAULT_NUM_WORKERS))
    c_param = float(os.environ.get("C_PARAM", DEFAULT_C_PARAM))
    logging_level = os.environ.get("LOGGING_LEVEL", DEFAULT_LOGGING_LEVEL)

    print(f"Initialized app with logging level {logging_level} and time budget {time_budget}.")

    logging.basicConfig(level=logging_level)

    app = PlayerServiceApp(FLASK_PACKAGE_NAME)

    app.add_player("random", AgentRandomSchieber())
    # no need for fallback because flask service will handle that and return internal server error
    # and the Jass Server will then use a fallback player (random) to play for me.
    # Also, can ignore_same_player_safety as the implementation is now state-free and tested
    # to ensure there aren't any cross-player issues. This removes the need for MultiPlayerAgentContainer.
    app.add_player(
        "ISMCTS",
        MultiProcessingISMCTS(
            time_budget,
            ucb1_c_param=c_param,
            num_workers=num_workers,
            ignore_same_player_safety=True,
        ),
    )

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8888)
