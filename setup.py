#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

MAJOR = 0
MINOR = 5
MICRO = 1
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

setup(
    name='pyinduct',
    version=VERSION,
    description="Toolbox for control and observer design for "
                "infinite dimensional systems.",
    long_description=readme[21:],
    long_description_content_type="text/x-rst",
    author="Stefan Ecklebe, Marcus Riesmeier",
    author_email='stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at',
    url='https://github.com/pyinduct/pyinduct/',
    download_url='https://github.com/pyinduct/pyinduct/releases',
    project_urls={
        "Documentation": "https://pyinduct.readthedocs.org",
        "Source Code": "https://github.com/pyinduct/pyinduct/",
        "Bug Tracker": "https://github.com/pyinduct/pyinduct/issues",
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="BSD 3-Clause License",
    zip_safe=False,
    keywords='distributed-parameter-systems control observer simulation',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    test_suite='pyinduct.tests',
    extras_require={
        "tests": ["codecov>=2.0.15"],
        "docs": ["sphinx>=3.2.1", "sphinx-rtd-theme>=0.5.0"],
    },
    python_requires=">=3.5",
)
