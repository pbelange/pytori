# copyright ############################### #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from setuptools import setup, find_packages, Extension
from pathlib import Path

#######################################
# Prepare list of compiled extensions #
#######################################

extensions = []

# LOAD REAME as PyPI description
with open("README.md","r") as fh:
    readme = fh.read()

#########
# Setup #
#########

version_file = Path(__file__).parent / 'pytori/_version.py'
dd = {}
with open(version_file.absolute(), 'r') as fp:
    exec(fp.read(), dd)
__version__ = dd['__version__']

setup(
    name='pytori',
    version=__version__,
    description='',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=extensions,
    include_package_data=True,
    install_requires=[
        'numpy>=1.0',
        'pandas',
        ],
    author='P. Belanger ',
    license='Apache 2.0',
    extras_require={
        'tests': ['pytest'],
        },
    )
