from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="imodels",
    version="0.3.0",
    author="Chandan Singh, Keyan Nasseri, Bin Yu, and others",
    author_email="chandan_singh@berkeley.edu",
    description="Implementations of various interpretable models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/imodels",
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        'corels==1.1.29',  # we only provide a basic wrapper around corels
        'cvxpy',
        'cvxopt',
        'mlxtend>=0.18.0',  # some lower version are missing fpgrowth
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn>=0.23.0',  # 0.23+ only works on py3.6+
    ],
    extras_require={
        'dev': [
            'dvu',
            'gdown',
            'jupyter',
            'jupytext',
            'matplotlib',
            'pytest',
            'pytest-cov',
            'slurmpy',
            'tqdm',
        ]
    },
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
