from distutils.core import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('VERSION.txt', 'r') as fh:
    version = fh.read().strip()

setup(
    name='aidefender',
    version=version,
    packages=find_packages(),
    url="https://dev.azure.com/MAIDAP/AI%20Defender/_git/aidefender",
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=requirements
)
