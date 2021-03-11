import setuptools
from os import path

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()
    
setuptools.setup(
    name="imodels",
    version="0.2.8",
    author="Chandan Singh, Keyan Nasseri, Bin Yu, and others",
    author_email="chandan_singh@berkeley.edu",
    description="Implementations of various interpretable models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/imodels",
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'scikit-learn>=0.23.0', # 0.23+ only works on py3.6+
        'cvxpy',
        'cvxopt',
        'mlxtend',
        'pytest',
        'pytest-cov',
        'tqdm'
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
