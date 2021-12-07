.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/pydeltasnow.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/pydeltasnow
    .. image:: https://readthedocs.org/projects/pydeltasnow/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://pydeltasnow.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/pydeltasnow/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/pydeltasnow
    .. image:: https://img.shields.io/pypi/v/pydeltasnow.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/pydeltasnow/
    .. image:: https://img.shields.io/conda/vn/conda-forge/pydeltasnow.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/pydeltasnow
    .. image:: https://pepy.tech/badge/pydeltasnow/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/pydeltasnow
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/pydeltasnow

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===========
pydeltasnow
===========


    Reimplementation of the delta.snow model by Winkler et al. 2021: Snow water equivalents exclusively from snow depths and their temporal changes.


A longer description of your project goes here...


.. _pyscaffold-notes:

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd pydeltasnow
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

Don't forget to tell your contributors to also install and use pre-commit.

.. _pre-commit: https://pre-commit.com/

Note
====

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
