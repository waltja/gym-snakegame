from setuptools import setup, find_packages

setup(
    name="gym-snakegame",
    version="0.1.0",
    author="helpingstar",
    author_email="iamhelpingstar@gmail.com",
    description="A gymnasium-based RL environment for learning the snake game.",
    packages=find_packages(),
    license="MIT License",
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pygame",
        "gymnasium",
        "moviepy",
        "ray",
    ],
)
