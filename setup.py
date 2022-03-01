from distutils.core import setup
from platform import platform

from setuptools import find_packages



setup(
    name='rl',
    version=1.0,
    install_requires=[
        'cloudpickle',
        'matplotlib>=3.4.3',
        'numpy>=1.21.2',
        'torch>=1.9.0',
        'wandb>=0.12.10',
        'gym==0.20.0',
        'mujoco-py==2.0.2.13'
        # 'mujoco-py>=2.0'
    ],
    description="RL-AMMI tools for combining deep RL algorithms.",
    authors="AAMI: MohamedElfatih Salah, Rami Ahmed*, Ruba Mutasim, Wafaa Mohammed",
    url="https://github.com/RamiSketcher/AMMI-RL",
    author_email="*rahmed@aimsammi.com"
)
