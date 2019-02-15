from setuptools import setup, find_packages
setup(
    name='bez2018model',
    version='0.1',
    packages=find_packages(),
    package_data={'': ['BEZ2018model/*']},
    include_package_data=True,
    install_requires=[],
)