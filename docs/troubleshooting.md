In case you run into issues with installation, here are some things that could help:

If you don't have permissions to install on your machine, use the --user flag:

`pip install git+https://github.com/csinva/imodels --user`


To test if everything is successfully installed, just try importing imodels from python:

```
python
> import imodels
```

To run example notebooks and/or develop locally, clone the repo then run

`pip install -e .[dev]`
