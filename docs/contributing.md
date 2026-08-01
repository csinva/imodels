We love open-source contributions and they are very welcome 🤗!

We have a queue of open things we are working on [here](https://github.com/csinva/imodels/projects/1) and in the open [issues](https://github.com/csinva/imodels/issues). Feel free to open an issue or contact @csinva (cs1@berkeley.edu or https://www.linkedin.com/in/csinva/) if you want to contribute!

Before contributing, it would be good to read the sklearn estimator [contributing guide](https://scikit-learn.org/stable/developers/develop.html) and generally be familiar with sklearn.
- For examples, functions/classes that are not meant for external use should start with an underscore (e.g. `_Rule`)  

[Docs](https://csinva.io/imodels/docs/) are built using [pdoc](https://pdoc3.github.io/pdoc/). Build them by changing to the `docs` directory and then running `./build_docs.sh`.

## Tests

[Tests](tests) are run with [pytest](https://docs.pytest.org/en/stable/) - run `pytest` in the repo directory (or `uv run pytest tests`, which is what CI runs). The full suite takes about 20 seconds. Make sure it passes before pushing code. Note that you might need to install some additional dependencies in order to get the tests to pass.

The tests are organized as:

- `tests/model_api_test.py` - runs **every** model registered in `imodels.CLASSIFIERS` / `imodels.REGRESSORS` through the conventions that should hold for all of them: `fit` returns `self`, `predict` / `predict_proba` shapes, probabilities summing to 1, DataFrame input, string class labels, refitting, sklearn `clone` and `get_params`/`set_params`, and `repr` before and after fitting.
- `tests/model_configs.py` - the small, fast settings each model is tested with.
- the remaining `*_test.py` files - checks specific to a single model or utility.

### Adding a new model

1. Export it from `imodels/__init__.py` and add it to `CLASSIFIERS` or `REGRESSORS`.
2. Add an entry to `MODEL_KWARGS` in `tests/model_configs.py` if the defaults are slow (keep each model well under a second), and to `BINARY_INPUT_MODELS` if it needs pre-discretized features.

The shared suite then covers the new model automatically. `TestRegistryCoverage` fails if a registered model has not been accounted for, so a model can't silently go untested. If a model genuinely can't satisfy the shared contract, add it to `EXCLUDED_MODELS` with a reason and cover it in its own test file, rather than weakening the shared checks.

Tests should not depend on the order they run in. `tests/conftest.py` seeds the global numpy/random state before each test, since some models draw from it (and some reseed it during `fit`).

The model is on [pypi](https://pypi.org/project/imodels/). Packaged following [this tutorial](https://realpython.com/pypi-publish-python-package/). Relevant commands:
```bash
uv build
uv publish
```


## Tutorials

Some models, e.g. [FIGS](https://csinva.io/imodels/figs.html) and [hierarchical shrinkage](https://csinva.io/imodels/shrinkage.html) have their own dedicated doc pages.

To add a doc page like this, copy `docs/figs.html` into a new file and then add in the relevant content. You will also need to manually edit the TOC under "Our favorite models" of each of the `html.mako` file in this repo (and other existing tutorials).

You may also need to clean up a string in `style_docs.py`...
