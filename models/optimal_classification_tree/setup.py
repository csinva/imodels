#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='pyoptree',
    version="1.0.3",
    description=(
        "Python Implementation of Bertsimas's paper Optimal Classification Trees."
    ),
    long_description=open('README.md').read(),
    author='Meng Pan',
    author_email='meng.pan95@gmail.com',
    maintainer='Meng Pan',
    maintainer_email='meng.pan95@gmail.com',
    license='BSD License',
    packages=find_packages(),
    platforms=["all"],
    url='https://github.com/pan5431333/pyoptree',
    install_requires=[
        'numpy>=1.14.5',
        'pandas>=0.23.1',
        'pyomo>=5.5.0',
        'scikit-learn>=0.20.0',
        'tqdm>=4.26.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
)
