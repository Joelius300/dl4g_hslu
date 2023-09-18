import sys
print(f"sys.path from {__name__}: {sys.path}")

from agents.myagent import *
from arena.myarena import *

print("hello after internal imports")

from jass.game.game_observation import GameObservation

print("hello after external imports")
