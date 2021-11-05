from os import path

import platform
import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'corels==1.1.29',  # we only provide a basic wrapper around corels
    # optionally requires cvxpy for slim
    'mlxtend>=0.18.0',  # some lower version are missing fpgrowth
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn>=0.23.0',  # 0.23+ only works on py3.6+
]
excluded_dirs = ['imodels.tests', 'experiments', 'notebooks', 'docs', 'imodels.tree.gosdt']

# gosdt is only supported on x86 64-bit systems
if 'x86_64' in platform.platform() and platform.system() != 'Windows':
    required_pypi.append('gosdt')
    excluded_dirs.pop()

setuptools.setup(
    name="imodels",
    version="1.1.0",
    author="Chandan Singh, Keyan Nasseri, Bin Yu, and others",
    author_email="chandan_singh@berkeley.edu",
    description="Implementations of various interpretable models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/imodels",
    packages=setuptools.find_packages(exclude=excluded_dirs),
    install_requires=required_pypi,
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
