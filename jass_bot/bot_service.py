import logging
import os

from jass_bot.agents.CompositeAgent import CompositeAgent
from jass_bot.agents.MultiPlayerAgentContainer import MultiPlayerAgentContainer
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.service.player_service_app import PlayerServiceApp
from jass_bot.agents.ISMCTS import ISMCTSCardStrategy

DEFAULT_TIME_BUDGET = 5
DEFAULT_LOGGING_LEVEL = logging.WARNING
FLASK_PACKAGE_NAME = "jass_bot"  # name of the package that contains the code for the flask application


def create_app():
    time_budget = float(os.environ.get("TIME_BUDGET", DEFAULT_TIME_BUDGET))
    logging_level = os.environ.get("LOGGING_LEVEL", DEFAULT_LOGGING_LEVEL)

    print(f"Initialized app with logging level {logging_level} and time budget {time_budget}.")

    logging.basicConfig(level=logging_level)

    app = PlayerServiceApp(FLASK_PACKAGE_NAME)

    app.add_player("random", AgentRandomSchieber())
    # TODO add random fallback
    app.add_player("ISMCTS", MultiPlayerAgentContainer(lambda: ISMCTS(time_budget)))
    # app.add_player("ISMCTS", MultiPlayerAgentContainer(lambda: ISMCTS(time_budget)))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8888)
