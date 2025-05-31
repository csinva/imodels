from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'matplotlib',
    'mlxtend>=0.18.0',  # some lower versions are missing fpgrowth
    'numpy',
    # tested with pandas 2.2.2 (but installing this pandas version will try to use newer np versions)
    'pandas',
    'requests',  # used in c4.5
    'scipy',
    'scikit-learn<1.6.0',  # 1.6.0 has issue with ensemble models
    'tqdm',  # used in BART
]

extra_deps = [
    'cvxpy',  # optionally requires cvxpy for slim
    'corels',  # optionally requires corels for optimalrulelistclassifier
    'gosdt-deprecated',  # optionally requires gosdt for optimaltreeclassifier
    'irf',  # optionally require irf for iterativeRandomForestClassifier
]

setuptools.setup(
    name="imodels",
    version="2.0.0",
    author="Chandan Singh, Keyan Nasseri, Matthew Epland, Yan Shuo Tan, Omer Ronen, Tiffany Tang, Abhineet Agarwal, Theo Saarinen, Bin Yu, and others",
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
    python_requires='>=3.9.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
