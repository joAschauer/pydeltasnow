.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

.. image:: https://readthedocs.org/projects/pydeltasnow/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://pydeltasnow.readthedocs.io/en/stable/
.. image:: https://img.shields.io/pypi/v/pydeltasnow.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/pydeltasnow/
.. image:: https://github.com/joAschauer/pydeltasnow/actions/workflows/tests.yml/badge.svg
    :alt: GitHub Workflow Status
    :target: https://github.com/joAschauer/pydeltasnow/actions/workflows/tests.yml


===========
pydeltasnow
===========


Python implementation of the deltaSNOW model by Winkler et al. 2021:

Winkler, M., Schellander, H., and Gruber, S.: Snow water equivalents
exclusively from snow depths and their temporal changes: the DeltaSNOW model,
Hydrol. Earth Syst. Sci., 25, 1165-1187, 
`10.5194/hess-25-1165-2021 <https://doi.org/10.5194/hess-25-1165-2021>`_, 2021.

The original implementation is included within the nixmass_ R package
of Winkler et al 2021. Differences of this version to the original R 
version are the following:

* The model accepts breaks in the date series if a break is sourrounded
  by zeros. Additionally, breaks in the date series can be accepted if
  surrounded by NaNs. See below for more information. This behaviour
  can be useful for measurement series that are not continued in summer.
* The user can specify how to deal with missing values in a measurement
  series. There are three parameters that control NaN handling:

  * ``ignore_zeropadded_gaps``
  * ``ignore_zerofollowed_gaps``
  * ``interpolate_small_gaps``

  Note that the runtime efficiency of the model will decrease when one
  or several of these options are turnded on.
* Accepts as input data only a pd.Series with pd.DatetimeIndex and no
  dataframe.
* The time resolution (timestep in R implementation) will be automatically
  sniffed from the DatetimeIndex of the input series.
* The user can specify the input and output units of the HS and SWE
  measurement series, respectively.
* A pd.Series with the dates as pd.DatetimeIndex is returned.


The core of this code is mainly based `on the work of Manuel Theurl
<https://github.com/manueltheurl/snow_to_swe>`_, this version makes use of the
numba_ just-in-time compiler for performance optimization.


Dependencies
============

``pydeltasnow`` depends on the following packages:

* pandas_: >=1.3
* numpy_: <1.21, >=1.17
* numba_: >=0.53.1

.. _installation:

Installation
============
Install ``pydeltasnow`` and its dependencies by runnig::

    pip install pydeltasnow


Usage
=====

For examples on how to use the package, please have a look at the
documentation_ of this project.


.. _documentation: https://pydeltasnow.readthedocs.io/en/stable/
.. _numba: https://numba.pydata.org/
.. _numpy: https://numpy.org/
.. _nixmass: https://CRAN.R-project.org/package=nixmass
.. _pandas: https://pandas.pydata.org/
