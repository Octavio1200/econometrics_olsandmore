from setuptools import setup, find_packages

setup(
    name='econometrics_olsandmore',
    version='0.1.0',
    description='Librería de econometría con OLS y más',
    author='Tu Nombre',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

import setuptools
setuptools.dist.Distribution().fetch_build_eggs = lambda *args, **kwargs: []
