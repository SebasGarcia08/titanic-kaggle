from distutils.core import setup
from setuptools import find_packages
from typing import List

packages_list: List[str] = [
    f"titanic.{s}" for s in find_packages(where="titanic")]
print("Packages to install")
print(packages_list)

dependencies = []

with open('dev_requirements.txt') as f:
    requirements = f.read().splitlines()
    for line in requirements:
        if len(line) > 0:
            req = line.strip()
            if req[0].isalpha():
                dependencies.append(req)

setup(
    name='titanic',
    version='0.0.0',
    install_requires=dependencies,
    packages=packages_list,
    python_requires=">=3.7.0,<=3.9.0",
    long_description=open('README.md').read(),
)
