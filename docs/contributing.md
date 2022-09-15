We love open-source contributions and they are very welcome 🤗!

We have a queue of open things we are working on [here](https://github.com/csinva/imodels/projects/1) and in the open [issues](https://github.com/csinva/imodels/issues). Feel free to open an issue or contact @csinva (cs1@berkeley.edu or https://www.linkedin.com/in/csinva/) if you want to contribute!

Before contributing, it would be good to read the sklearn estimator [contributing guide](https://scikit-learn.org/stable/developers/develop.html) and generally be familiar with sklearn.
- For examples, functions/classes that are not meant for external use should start with an underscore (e.g. `_Rule`)  

[Docs](https://csinva.io/imodels/docs/) are built using [pdoc](https://pdoc3.github.io/pdoc/). Build them by changing to the `docs` directory and then running `./build_docs.sh`.

[Tests](tests) are run with [pytest](https://docs.pytest.org/en/stable/) (e.g. run `pytest` in the repo directory) - make sure they pass before pushing code, and that new models pass a reasonable set of tests. Note that you might need to install some additional dependencies in order to get the tests to pass.

The model is on [pypi](https://pypi.org/project/imodels/). Packaged following [this tutorial](https://realpython.com/pypi-publish-python-package/). Relevant commands:
```bash
pip install twine wheel

rm -rf build dist

python setup.py sdist bdist_wheel

twine check dist/*

twine upload dist/*
```


## Tutorials

Some models, e.g. [FIGS](https://csinva.io/imodels/figs.html) and [hierarchical shrinkage](https://csinva.io/imodels/shrinkage.html) have their own dedicated doc pages.

To add a doc page like this, copy `docs/figs.html` into a new file and then add in the relevant content. You will also need to manually edit the TOC under "Our favorite models" of each of the `html.mako` file in this repo (and other existing tutorials).

You may also need to clean up a string in `style_docs.py`...
