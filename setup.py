from setuptools import setup, find_packages

setup(
    name="fluidplanner",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "tensorboard",
    ],
)
