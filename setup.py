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
    name = "PyGraphStat",
    version = "0.1",
    packages = find_packages(),
    py_modules = (['PyGraphstat',
					'PyGraphStat.RandomGraph', 
					'PyGraphStat.tests', 
					'PyGraphStat.embedding',
					'PyGraphStat.classifiers',
					'PyGraphStat.hypothesis-testing']),
    install_requires = ['numpy>=1.3', 'scipy>=0.7', 'matplotlib>=0.99', 'h5py>=1.3', 'multiprocessing>=0.7', 'networkx>=1.6'],

    # metadata for upload to PyPI
    author = "Joshua T. Vogelstein, Logan Grosenick, and Daniel Sussman",
    author_email = "joshuav@jhu.edu",
    description = 'statistical inference on graphs',
    long_description=read('README'),
    license = "CC BY 3.0",
    keywords = "graph statistics",
    url = "https://github.com/jovo/PyGraphStat", 

    classifiers=[
        "Development Status :: Alpha",
    ],
)
