from setuptools import setup

setup(
    name="torchrl_mcts",
    version="1.0",
    description="MCTS for torchrl",
    author="Majid Laali",
    author_email="mjlaali@gmail.com",
    packages=["mcts"],
    install_requires=["torchrl"],
)
