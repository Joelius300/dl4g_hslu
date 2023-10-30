import logging

from jass_bot.agents.ISMCTS import ISMCTS


def create_app():
    from jass.agents.agent_random_schieber import AgentRandomSchieber
    from jass.service.player_service_app import PlayerServiceApp

    logging.basicConfig(level=logging.DEBUG)

    app = PlayerServiceApp("player_service")

    app.add_player("random", AgentRandomSchieber())
    app.add_player("ISMCTS", ISMCTS(timebudget=.1))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8888)
