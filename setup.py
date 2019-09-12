from setuptools import setup
from setuptools import find_packages

setup(
    name = 'ctree',
    version = '0.1',
     description = 'Implements the causal estimators descibed in https://arxiv.org/pdf/1504.01132.pdf',
    author = 'Kristian Urup Olesen Larsen',
    packages = find_packages(),
    install_requires = ['numpy', 'pandas', 'scikit-learn', 'pygraphviz', 'networkx', 'numba']
)