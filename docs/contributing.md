Contributions are very welcome!

We have a queue of open things we are working on [here](https://github.com/csinva/imodels/projects/1). Feel free to open an issue or contact @csinva (cs1@berkeley.edu or https://www.linkedin.com/in/csinva/) if you want to contribute!

Before contributing, it would be good to read the sklearn estimator [contributing guide](https://scikit-learn.org/stable/developers/develop.html) and generally be familiar with sklearn. Also please ensure that any added implementations use the appropriate classes from `imodels.util` (e.g. the `Rule` class). 

[Docs](https://csinva.io/imodels/docs/) are built using [pdoc](https://pdoc3.github.io/pdoc/). Build them by changing to the `docs` directory and then running `build_docs.sh`.

[Tests](tests) are run with [pytest](https://docs.pytest.org/en/stable/) (e.g. run `pytest` in the repo directory) - make sure they pass before pushing code, and that new models pass a reasonable set of tests. Note that you might need to install some additional dependencies in order to get the tests to pass.

Project is also on [pypi](https://pypi.org/project/imodels/).
