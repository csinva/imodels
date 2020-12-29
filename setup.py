import setuptools

setuptools.setup(
    name="imodels",
    version="0.2.5",
    author="Chandan Singh",
    author_email="chandan_singh@berkeley.edu",
    description="Implementations of various interpretable models",
    long_description="Interpretable ML package for concise, transparent, and accurate predictive modeling (sklearn-compatible).",
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/imodels",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy<1.20',
        'scipy<1.6',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'cvxpy',
        'cvxopt',
        'mlxtend',
        'pytest',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
