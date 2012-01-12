#!/usr/bin/env python

# Imports
import os, sys, string
from setuptools import setup, find_packages

#-----------------------------------------------------------------------------#
# Functions

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

#-----------------------------------------------------------------------------#
# setup call

setup(
    name = "PyGraphStats",
    version = "0.1",
    packages = find_packages(),
    py_modules = ['random', 'embedding', 'classifiers',],
    install_requires = ['numpy>=1.3', 'scipy>=0.7', 'matplotlib>=0.99', 'h5py>=1.3', 'multiprocessing>=0.7'],

    # metadata for upload to PyPI
    author = "Joshua T. Vogelstein, Logan Grosenick, and Daniel Sussman",
    author_email = "joshuav@jhu.edu",
    description = '''This package provides the ability to perform statistical inference on graphs 
		We provide methods to perform hypothesis testing, graph classification, unsupervised vertex clustering 
		and semi-supervised vertex clustering. We also provide embedding techniques that give a representation 
		of the graph in Euclidean space. Additionally we provide methods to generate random graphs such as stochastic 
		blockmodel graphs and random dot product graphs.''',
    long_description=read('README'),
    license = "CC",
    keywords = "graph statistics",
    url = "https://github.com/jovo/PyGraphStat", 

    classifiers=[
        "Development Status :: Alpha",
    ],
)
