import setuptools

setuptools.setup(
    name="imodels",
    version="0.2.3",
    author="Chandan Singh",
    author_email="chandan_singh@berkeley.edu",
    description="Implementations of various interpretable models",
    long_description="Implementations of various interpretable models",
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/interpretability-implementations-demos",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'cvxpy',
#         'fim @ git+https://github.com/csinva/pyfim-clone',
#         'irf @ git+https://github.com/Yu-Group/iterative-Random-Forest',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
