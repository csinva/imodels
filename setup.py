from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'matplotlib',
    'mlxtend>=0.18.0',  # some lower version are missing fpgrowth
    'numpy',
    'pandas',
    'requests',  # used in c4.5
    'scipy',
    'scikit-learn',  # 0.23+ only works on py3.6+
    'tqdm',  # used in BART
]

extra_deps = [
    'cvxpy',  # optionally requires cvxpy for slim
    'corels',  # optinally requires corels for optimalrulelistclassifier
    'gosdt-deprecated',  # optionally requires gosdt for optimaltreeclassifier
    'irf',  # optionally require irf for iterativeRandomForestClassifier
]

setuptools.setup(
    name="imodels",
    version="1.3.0",
    author="Chandan Singh, Keyan Nasseri, Bin Yu, and others",
    author_email="chandan_singh@berkeley.edu",
    description="Implementations of various interpretable models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/imodels",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    install_requires=required_pypi,
    extras_require={
        'dev': [
            'dvu',
            'gdown',
            # 'irf',
            'jupyter',
            'jupytext',
            'matplotlib',
            # 'pdoc3',  # for building docs
            'pytest',
            'pytest-cov',
            # 'seaborn',  # in bartpy.diagnostics.features
            'slurmpy',
            # 'statsmodels', # in bartpy.diagnostics.diagnostics
            # 'torch',  # for neural-net-integrated models
            'tqdm',
            'pmlb',
        ]
    },
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
