from setuptools import find_packages, setup

setup(
    name='grocerystore',
    packages=find_packages("src"),
    package_dir={"": "src"},
    version='0.1.0',
    description='This project uses simpy to simulate a basic grocery store model',
    author='Rachel Kalusniak',
    license='',
)
