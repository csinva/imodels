In case you run into issues with installation, here are some things that could help:

If you don't have permissions to install on your machine, use the --user flag:

`pip install git+https://github.com/csinva/imodels --user`

If you are running into an issue installing fim or irf, you may have to install these dependencies individually. In that case, these commands will install everything:

```python
pip install --upgrade pip
pip install pytest numpy scipy matplotlib pandas scikit-learn cvxpy
pip install git+https://github.com/csinva/pyfim-clone
pip install git+https://github.com/Yu-Group/iterative-Random-Forest
```

To test if everything is successfully installed, just try importing imodels from python:

```
python
> import imodels
```

To develop locally, clone the repo then run

`python setup.py develop`
