==================
Examples and Usage
==================

This section shows briefly how to use the ``pydeltasnow`` package. Before you can
start, please install the package as described in the :ref:`installation` section.

After installing you can import the model alongside with pandas, which is 
necessary since ``pydeltasnow`` relies on pandas objects as input data::

    from pydeltasnow import swe_deltasnow
    import pandas as pd

Then you can access and calculate the snow water equivalent (SWE) with the
DeltaSNOW model via::

    hs_series = pd.read_csv("path/to/some/hs_data.csv",
                          parse_dates=['date'],
                          index_col='date').squeeze()

    swe = swe_deltasnow(hs_series)

where ``hs_data`` is a :class:`pandas.Series` with the snow depth (HS) data and
a :class:`pandas.DatetimeIndex`. The variable ``swe`` will also be a
:class:`pandas.Series` with the same index as the input data.

If you want to add a new column of the modeled SWE to an existing
:class:`pandas.DataFrame` which also holds the HS data, you can use the 
:func:`pandas.DataFrame.assign` operator to add a new column. Assume we load 
data from a .csv table in which the column ``hs`` holds the snow depth in [cm]
and we want add a swe column in [m]::

    data = (pd.read_csv("path/to/some/hs_data.csv",
                        parse_dates=['date'],
                        index_col='date')
            .assign(swe_in_m=lambda x: swe_deltasnow(x['hs'],
                                                     hs_input_unit='cm'
                                                     swe_output_unit='m')
            )

By default, the model uses the optimized parameters from the `DeltaSNOW paper 
<https://doi.org/10.5194/hess-25-1165-2021>`_. You can change these parameters
by overwriting the default parameters in the
:func:`pydeltasnow.main.swe_deltasnow` function::

    swe = swe_deltasnow(hs_series,
                        rho_max=450)

Please read the paper on how the parameters influence the behaviour of the 
model and what potential pitfalls are.

If your input data contains gaps that are irrelavant for you (i.e gaps that are
surrounded or followed by zeros), you can ignore these gaps::
    
    swe = swe_deltasnow(hs_series,
                        ignore_zerofollowed_gaps=True)

Otherwise, :func:`pydeltasnow.main.swe_deltasnow` will not accept any missing
data in your input series. Additionally, you can interpolate gaps that are 
shorter or equally long as ``max_gap_length`` if you set 
``interpolate_small_gaps`` to ``True``.