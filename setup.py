'''
Setup Script for the Recommender Module

This will install the ask_watson module to the local python distribution.
'''


from setuptools import setup
from setuptools import find_packages


__status__      = "Package"
__copyright__   = "Copyright 2018"
__license__     = "MIT License"
__version__     = "1.0.0"

# 01101100 00110000 00110000 01110000
__author__      = "Felix Geilert"


def readme():
    '''Retrieves the content of the readme.'''
    with open('readme.md') as f:
        return f.read()


setup(name='ask_watson',
      version=__version__,
      description='Sklearn Extension to integration recommender functions',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='recommender systems',
      url='https://github.com/felixnext/ds_recommender',
      author='Felix Geilert',
      license='MIT License',
      packages=find_packages(),
      install_requires=[ 'numpy', 'sklearn', 'scipy', 'pandas' ],
      include_package_data=True,
      zip_safe=False)
