In case you run into issues with installation, here are some things that could help:

If you don't have permissions to install on your machine, use the --user flag:

`pip install git+https://github.com/csinva/imodels --user`

Note that some models (e.g. the ones below) require extra dependencies:

```python
extra_deps = [
    'cvxpy',  # optionally requires cvxpy for slim
    'corels',  # optionally requires corels for optimalrulelistclassifier
    'gosdt',  # optionally requires gosdt for optimaltreeclassifier
    'irf',  # optionally require irf for iterativeRandomForestClassifier
]
```


To test if everything is successfully installed, just try importing imodels from python:

```
python
> import imodels
```

To run example notebooks and/or develop locally, clone the repo then run

`pip install -e ".[dev]"`
