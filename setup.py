import setuptools

setuptools.setup(
    name="imodels",
    version="0.2.7",
    author="Chandan Singh",
    author_email="chandan_singh@berkeley.edu",
    description="Implementations of various interpretable models",
    long_description="Interpretable ML package for concise, transparent, and accurate predictive modeling (sklearn-compatible).",
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/imodels",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'scikit-learn>=0.23.0', # 0.23+ only works on py3.6+)
        'cvxpy',
        'cvxopt',
        'mlxtend',
        'pytest',
        'pytest-cov'
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
