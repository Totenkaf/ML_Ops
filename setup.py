"""Copyright 2022 by Artem Ustsov"""

from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="Homework_1",
    packages=find_packages(),
    version="0.1.0",
    description="",
    author="Production ready project",
    entry_points={
        "console_scripts": [
            "ml_project_train = ml_project.train_pipeline:train_pipeline_command"
        ]
    },
    install_requires=required,
    license="MIT",
)
