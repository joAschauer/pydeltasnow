==================
Examples and Usage
==================

This section shows briefly how to use the ``pydeltasnow`` package. Before you can
start, please install the package as described in the :ref:`installation` section.

After installing you can import the package with::

    import pydeltasnow

Then you can access and calculate the snow wter equivalent (SWE) with the DeltaSNOW model via::

    swe = pydeltasnow.swe_deltasnow(hs_data)

where ``hs_data`` is a :class:`pandas.Series` with the snow depth (HS) data and
a :class:`pandas.DatetimeIndex`. The variable ``swe`` will also be a
:class:`pandas.Series` with the same index as the input data.

If you want to add a new column of the modeled SWE to an existing
:class:`pandas.DataFrame` which also holds the HS data, you can use the 
:func:`pandas.DataFrame.assign` operator to add a new column. Assume we load 
data from a .csv file in which the column ``hs`` holds the snow depth in [cm]
and we want add a swe column in [m]::

    data = (pd.read_csv("path/to/some/hs_data.csv",
                        parse_dates=['date'],
                        index_col='date')
            .assign(swe_in_m=lambda x: swe_deltasnow(x['hs'],
                                                     hs_input_unit='cm'
                                                     swe_output_unit='m')
            )