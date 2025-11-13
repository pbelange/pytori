# copyright ############################### #
# Copyright (c) P. Belanger, 2025.          #
# ######################################### #

from setuptools import setup, find_packages
from pathlib import Path

#######################################
# Prepare list of compiled extensions #
#######################################
extensions = []

# LOAD README as PyPI description
with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

#########
# Setup #
#########

version_file = Path(__file__).parent / "pytori/_version.py"
dd = {}
with open(version_file, "r") as fp:
    exec(fp.read(), dd)
__version__ = dd["__version__"]

setup(
    name="pytori",
    version=__version__,
    description="Transport of loops and tori in accelerator beam lines",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/pbelange/pytori",
    author="P. Belanger",
    license="Apache-2.0",
    packages=find_packages(),
    ext_modules=extensions,
    include_package_data=True,
    install_requires=[
        "numpy>=1.0",
        "pandas",
    ],
    extras_require={
        "tests": ["pytest"],
    },
    python_requires=">=3.8",
    project_urls={
        "Source Code": "https://github.com/pbelange/pytori",
        # "Bug Tracker": "https://github.com/pbelange/pytori/issues",
        # "Documentation": "https://pytori.readthedocs.io/",
    },
)
