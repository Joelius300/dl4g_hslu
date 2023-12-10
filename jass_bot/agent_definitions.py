from jass_bot.ML.trump_selection.model_trump_strategy import ModelTrumpStrategy
from jass_bot.agents.ISMCTS import ISMCTS
from jass_bot.heuristics import graf
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass_bot.strategies.card_strategy import CardStrategy
from jass_bot.strategies.trump_strategy import TrumpStrategy
from jass_bot.strategies.fallback_card_strategy import FallbackCardStrategy
from jass_bot.strategies.fallback_trump_strategy import FallbackTrumpStrategy


def _just_fail(_obs):
    raise Exception("Just fail :)")


def get_trump_strat(trump_def: dict, with_random_fallback=False) -> TrumpStrategy:
    name = trump_def["name"]
    if name == "random":
        strat = TrumpStrategy.from_agent(AgentRandomSchieber())
        with_random_fallback = False
    elif name == "graf":
        strat = TrumpStrategy.from_function(graf.graf_trump_selection)
    elif name == "model":
        strat = ModelTrumpStrategy(trump_def["checkpoint_path"])
    elif name == "just_fail":
        strat = TrumpStrategy.from_function(_just_fail)
    else:
        raise ValueError(f"Invalid trump strat {name}")

    if not with_random_fallback:
        return strat

    return FallbackTrumpStrategy(
        strat, TrumpStrategy.from_agent(AgentRandomSchieber()), fallback_name="random"
    )


def get_card_strat(card_def: dict, with_random_fallback=False) -> CardStrategy:
    name = card_def["name"]
    if name == "random":
        strat = CardStrategy.from_agent(AgentRandomSchieber())
        with_random_fallback = False
    elif name == "ISMCTS":
        strat = CardStrategy.from_agent(ISMCTS(time_budget=card_def["time_budget"]))
    elif name == "just_fail":
        strat = CardStrategy.from_function(_just_fail)
    else:
        raise ValueError(f"Invalid card strat {name}")

    if not with_random_fallback:
        return strat

    return FallbackCardStrategy(
        strat, CardStrategy.from_agent(AgentRandomSchieber()), fallback_name="random"
    )


class CardDefs:
    @classmethod
    def random(cls):
        return dict(name="random")

    @classmethod
    def ISMCTS(cls, time_budget: float):
        return dict(name="ISMCTS", time_budget=time_budget)

    @classmethod
    def just_fail(cls):
        return dict(name="just_fail")


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

    @classmethod
    def just_fail(cls):
        return dict(name="just_fail")
