# Jass Bot DL4G HSLU

A bot that plays the swiss card game Jass. Implemented for the Deep Learning for Games module at HSLU.

## Running it

(subject to change obviously, needs to work with whatever deployment you'll be using)

```bash
python -m jass_bot.main
```

## Development

Install the dev dependencies as follows (in a venv obviously).
This also installs the jass_kit package (git submodule of the fork) as an editable package.

Also ensure that nbstripout is initialized correctly.

```bash
pip install -r requirements_dev.txt
nbstripout --install --attributes .gitattributes
```

### Note on editable package and pythons incompetence

This should work and allow you to edit the fork without having to reinstall all the time.
However, for deployment, you'll need to see if the `./jass_kit` in `requirements.txt` is sufficient.

This is btw the "correct" way to work on a separate package locally at the same time, instead of forcing yourself to use relative imports everywhere until you stumble into a situation where that's not possible anymore. It is also not correct to use `sys.path` to hack it. Some IDEs/LSPs like pylsp have seemingly no issues with this and can easily use "Go to definition" but other smaller ones like PyCharm apparently need you to specify that the other folder also contains source files, even though the editable installation is correctly listed in the packages tab.
Earlier it was not recommended to use editable installs because of poor support but some claim that this has changed. Anyhow, Python, wtf is wrong with you. I would not touch you with a ten foot pole if you weren't the most important language in AI/ML atm.

