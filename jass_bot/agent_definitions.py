from jass_bot.ML.trump_selection.model_trump_strategy import ModelTrumpStrategy
from jass_bot.agents.ISMCTS import ISMCTS
from jass_bot.heuristics import graf
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass_bot.strategies.card_strategy import CardStrategy
from jass_bot.strategies.trump_strategy import TrumpStrategy


def get_trump_strat(trump_def: dict) -> TrumpStrategy:
    name = trump_def["name"]
    if name == "random":
        return TrumpStrategy.from_agent(AgentRandomSchieber())
    elif name == "graf":
        return TrumpStrategy.from_function(graf.graf_trump_selection)
    elif name == "model":
        return ModelTrumpStrategy(trump_def["checkpoint_path"])


def get_card_strat(card_def: dict) -> CardStrategy:
    name = card_def["name"]
    if name == "random":
        return CardStrategy.from_agent(AgentRandomSchieber())
    elif name == "ISMCTS":
        return CardStrategy.from_agent(ISMCTS(time_budget=card_def["time_budget"]))


class CardDefs:
    @classmethod
    def random(cls):
        return dict(name="random")

    @classmethod
    def ISMCTS(cls, time_budget: float):
        return dict(name="ISMCTS", time_budget=time_budget)


class TrumpDefs:
    @classmethod
    def random(cls):
        return dict(name="random")

    @classmethod
    def graf(cls):
        return dict(name="graf")

    @classmethod
    def model(cls, checkpoint_path):
        return dict(name="model", checkpoint_path=checkpoint_path)
