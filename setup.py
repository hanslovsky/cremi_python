#!/usr/bin/env python

from distutils.core import setup

setup(
    name='cremi',
    version='0.7',
    description='Python Package for the CREMI Challenge',
    author='Jan Funke',
    author_email='jfunke@iri.upc.edu',
    url='http://github.com/funkey/cremi_python',
    packages=['cremi', 'cremi.io', 'cremi.evaluation'],
    entry_points={'console_scripts': [
        'eval-watersheds = cremi.eval_watersheds:run_evaluation',
        'eval-mutex-watersheds = cremi.eval_watersheds_mutex:run_evaluation']}
)
