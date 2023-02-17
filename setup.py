#!/usr/bin/env python
import os
import io
import re
import shutil
import sys

import setuptools
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
VERSION = find_version("pytvision", "__init__.py")
requirements = open("requirements.txt").read()

setup(
    # Metadata
    name="pytvision",
    version=VERSION,
    author="Pedro Diamel Marrero Fernandez",
    author_email="pedrodiamel@gmail.com",
    url="https://github.com/pedrodiamel/pytorchvision",
    description="Tranformers, synthetycs and visualization tools for pytorch deep learning framework",
    long_description=readme,
    license="MIT",
    # Package info
    packages=find_packages(exclude=("test",)),
    zip_safe=True,
    install_requires=requirements,
)
