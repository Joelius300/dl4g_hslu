# Jass Bot DL4G HSLU

A bot that plays the Swiss card game Jass using Information Set Monte Carlo Tree Search with Root Parallelization.
Implemented for the Deep Learning for Games (DL4G) module at HSLU. \
There's also a [short report](./report/Jass-Queen.pdf) in german if you want to read it.
Unfortunately, I didn't have time to implement Reinforcement Learning, but it still performed pretty well (4th place in tournament).

The two most interesting files for the bot are [`ISMCTS.py`](./jass_bot/agents/ISMCTS.py) and [`multi_processing_ISMCTS.py`](./jass_bot/agents/multi_processing_ISMCTS.py).
If you care about the machine learning approaches that were explored, see the [ML/trump_selection folder](./jass_bot/ML/trump_selection).

## Running it

The bot is using docker-compose and Gunicorn for hosting so running it is as simple as 

```bash
docker compose up
```

To run the bot locally (must set up all the environment variables first), run

```bash
python -m jass_bot.bot_service
```

Same for the main script, which is mainly used for experimentation.

```bash
python -m jass_bot.main
```

### PYTHONPATH

Because Python has a _great_ module resolution system, with no flaws whatsoever, you will want to execute the following command especially when working with external tools like DVC to set up a queue of experiments where you cannot always control from where the scripts will be called.


```zsh
export PYTHONPATH="$PWD"
```

## Development

Install the dev dependencies as follows (in a venv obviously).
This also installs the jass_kit package (git submodule of the fork) as an editable package.

Also ensure that nbstripout is initialized correctly.

```bash
pip install -r requirements_dev.txt
nbstripout --install --attributes .gitattributes
```
