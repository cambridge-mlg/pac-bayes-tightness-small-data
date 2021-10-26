from setuptools import setup, find_packages

requirements = [
    'numpy',
    'torch',
    'stheno',
    'matplotlib',
    'python-slugify',
    'tensorboard',
    'fdm',
    'backends',
    'plum-dispatch>=1.0',
    'pytest'
]

setup(
    name='pacbayes',
    version='0.0.1',
    packages=find_packages(),
    install_requires=requirements,
)
