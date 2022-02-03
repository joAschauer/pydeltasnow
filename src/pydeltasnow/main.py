# -*- coding: utf-8 -*-
"""
This module provides the main user interface to the deltaSNOW model with the
:func:`swe_delatsnow` function.

"""
import pandas as pd
import numpy as np
from numba import njit

from .core import deltasnow_snowpack_evolution

from .utils import (
    continuous_timedeltas_in_nonzero_chunks,
    fill_small_gaps,
    get_nonzero_chunk_idxs,
    get_zeropadded_gap_idxs,
    )

from pydeltasnow import __version__

__author__ = "Johannes Aschauer"
__copyright__ = "Johannes Aschauer"
__license__ = "GPL-2.0-or-later"


UNIT_FACTOR = {
    'mm': 0.001,
    'cm': 0.01,
    'm': 1.0,
    }


def _raise_nans_error_message(
    ignore_zeropadded_gaps,
    ignore_zerofollowed_gaps,
    interpolate_small_gaps,
    max_gap_length
):
    if (any([ignore_zeropadded_gaps, ignore_zerofollowed_gaps])
            and not interpolate_small_gaps):
        raise ValueError(("DeltaSNOW: your data contains NaNs surrounded "
                          "or followed by non-zeros."))
    elif (any([ignore_zeropadded_gaps, ignore_zerofollowed_gaps])
            and interpolate_small_gaps):
        raise ValueError(("DeltaSNOW: your data contains gaps of NaNs "
                          "that are either:\n"
                          "    - at the the end or beginning of your series\n"
                          f"    - longer than {max_gap_length} timesteps and "
                          "not surrounded or followed by nonzeros\n"
                          f"    - shorter than {max_gap_length} timestep(s) "
                          "but with breaks in the date index"))
    elif (interpolate_small_gaps 
            and not any([ignore_zeropadded_gaps, ignore_zerofollowed_gaps])):
        raise ValueError(("DeltaSNOW: your data contains gaps of NaNs "
                          "that are either:\n"
                          "    - at the the end or beginning of your series\n"
                          f"    - longer than {max_gap_length} timestep(s)\n"
                          f"    - shorter than {max_gap_length} timestep(s) "
                          "but with breaks in the date index"))
    else:
        raise ValueError("DeltaSNOW: snow depth data must not be NaN.")


@njit
def _deltasnow_on_nonzero_chunks(
    Hobs,
    swe_out,
    start_idxs,
    stop_idxs,
    rho_max,
    rho_null,
    c_ov,
    k_ov,
    k,
    tau,
    eta_null,
    resolution,
):
    """
    Model snowpack evolution on chunks of nonzeros in Hobs.

    Parameters
    ----------
    Hobs : 1D :class:`numpy.ndarray` of floats
        Measured snow depth. Needs to be in [m].
    swe_out : 1D :class:`numpy.ndarray` of floats
        preallocated swe array where the output is stored to. Same shape as
        `Hobs`.
    start_idxs : :class:`numpy.ndarray` of int
        Start indices of windows with nonzeros in `Hobs`.
    stop_idxs : :class:`numpy.ndarray` of int
        Last indices of windows with nonzeros in `Hobs`.
    rho_max : float, optional
        Maximum density of an individual snow layer produced by the DeltaSNOW
        model in [kg/m3], rho_max needs to be positive. The default is 401.2588.
    rho_null : float, optional
        Fresh snow density for a newly created layer [kg/m3], `rho_null` needs to
        be positive. The default is 81.19417.
    c_ov : float, optional
        Overburden factor due to fresh snow [-], `c_ov` needs to be positive. The
        default is 0.0005104722.
    k_ov : float, optional
        Defines the impact of the individual layer density on the compaction due
        to overburden [-], `k_ov` need to be in the range [0,1].
        The default is 0.37856737.
    k : float, optional
        Exponent of the exponential-law compaction [m3/kg], `k` needs to be
        positive. The default is 0.02993175.
    tau : float, optional
        Uncertainty bound [m], `tau` needs to be positive.
        The default is 0.02362476.
    eta_null : float, optional
        Effective compactive viscosity of snow for "zero-density" [Pa s].
        The default is 8523356.
    resolution : float
        Timedelta in hours between snow observations.

    Returns
    -------
    swe_out : 1D :class:`numpy.ndarray`

    """

    for start, stop in zip(start_idxs, stop_idxs):
        swe_out[start:stop] = deltasnow_snowpack_evolution(
            Hobs[start:stop],
            rho_max,
            rho_null,
            c_ov,
            k_ov,
            k,
            tau,
            eta_null,
            resolution,
            )

    return swe_out


def swe_deltasnow(
    data,
    rho_max=401.2588,
    rho_null=81.19417,
    c_ov=0.0005104722,
    k_ov=0.37856737,
    k=0.02993175,
    tau=0.02362476,
    eta_null=8523356.,
    hs_input_unit='m',
    swe_output_unit='mm',
    ignore_zeropadded_gaps=False,
    ignore_zerofollowed_gaps=False,
    interpolate_small_gaps=False,
    max_gap_length=3,
    interpolation_method='linear',
    output_series_name='swe_deltasnow',
):
    """
    Calculate snow water equivalent from a snow depth timeseries with the
    DeltaSNOW model.

    Parameters
    ----------
    data : :class:`pandas.Series` with :class:`pandas.DatetimeIndex`
        The input snow depth data. Needs to be numeric and not negative.
    rho_max : float, optional
        Maximum density of an individual snow layer produced by the DeltaSNOW
        model in [kg/m3], `rho_max` needs to be positive. The default is 401.2588.
    rho_null : float, optional
        Fresh snow density for a newly created layer [kg/m3], `rho_null` needs to
        be positive. The default is 81.19417.
    c_ov : float, optional
        Overburden factor due to fresh snow [-], `c_ov` needs to be positive. The
        default is 0.0005104722.
    k_ov : float, optional
        Defines the impact of the individual layer density on the compaction due
        to overburden [-], `k_ov` needs to be in the range [0,1].
        The default is 0.37856737.
    k : float, optional
        Exponent of the exponential-law compaction [m3/kg], `k` needs to be
        positive. The default is 0.02993175.
    tau : float, optional
        Uncertainty bound [m], `tau` needs to be positive.
        The default is 0.02362476.
    eta_null : float, optional
        Effective compactive viscosity of snow for "zero-density" [Pa s].
        The default is 8523356.
    hs_input_unit : str in {"mm", "cm", "m"}
        The unit of the input snow depth. The default is "m".
    swe_output_unit : str in {"mm", "cm", "m"}
        The unit of the output snow water equivalent. The default is "mm".
    ignore_zeropadded_gaps : bool
        Whether to ignore gaps that have leading and trailing zeros. The
        resulting SWE series will contain NaNs at the same positions. These
        gaps are also ignored when you use `ignore_zerofollowed_gaps`.
    ignore_zerofollowed_gaps : bool
        Less strict rule than `ignore_zeropadded_gaps`. Whether to ignore gaps
        that have trailing zeros. This can lead to sudden drops in SWE in case
        missing HS data is present. The resulting SWE series will contain NaNs
        at the same positions.
    interpolate_small_gaps : bool
        Whether to interpolate small gaps in the input HS data or not. Only gaps
        that are surrounded by data points and have continuous date spacing
        between the leading and trailing data point are interpolated.
    max_gap_length : int
        The maximum gap length of HS data gaps that are interpolated if
        `interpolate_small_gaps` is True.
    interpolation_method : str
        Interpolation method for the small gaps which is passed to
        :func:`pandas.Series.interpolate`. See its documentation for valid
        options. The default is 'linear'.
    output_series_name : str
        The name of the resulting pd.Series. This can be useful if you want to
        add the resulting SWE series to an existing DataFrame and need a
        specific column name. The default is "swe_deltasnow".

    Raises
    ------
    ValueError
        If any of the constraints on the data is violated.

    Returns
    -------
    swe : :class:`pandas.Series`
        Calculated SWE with the same index as the input data.
    
    Notes
    -----
    Differences to the original R implementation within the ``nixmass`` package
    of Winkler et al. 2021:

        - The model accepts breaks in the date series if a break is sourrounded
          by zeros. Additionally, breaks in the date series can be accepted if
          surrounded by NaNs. See below for more information. This behaviour
          can be useful for measurement series that are not continued in summer.
        - The user can specify how to deal with missing values in a measurement
          series. There are three parameters that control NaN handling:
            - ``ignore_zeropadded_gaps``
            - ``ignore_zerofollowed_gaps``
            - ``interpolate_small_gaps``
          Note that the runtime efficiency of the model will decrease when one
          or several of these options are turnded on.
        - Accepts as input data only a :class:`pandas.Series` with 
          :class:`pandas.DatetimeIndex` and no dataframe.
        - The time resolution (timestep in R implementation) will be 
          automatically sniffed from the :class:`pandas.DatetimeIndex` of the 
          input series.
        - The user can specify the input and output units of the HS and SWE
          measurement series, respectively.
        - A :class:`pandas.Series` with the dates as pd.DatetimeIndex is
          returned.

    """

    for unit in [hs_input_unit, swe_output_unit]:
        assert unit in UNIT_FACTOR.keys(), (f"swe.deltasnow: {unit} has to be "
                                            "in {'mm', 'cm', 'm'}")

    if not isinstance(data, pd.Series):
        raise ValueError("DeltaSNOW: data must be pd.Series")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DeltaSNOW: data needs pd.DatetimeIndex as index.")

    Hobs = data.mul(UNIT_FACTOR[hs_input_unit]).to_numpy()
    dates = data.index.to_numpy()

    if ignore_zeropadded_gaps or ignore_zerofollowed_gaps:
        if ignore_zerofollowed_gaps:
            zeropadded_gap_idxs = get_zeropadded_gap_idxs(
                Hobs,
                require_leading_zero=False)
        else:  # ignore_zeropadded_gaps with zero in front and back
            zeropadded_gap_idxs = get_zeropadded_gap_idxs(
                Hobs,
                require_leading_zero=True)
        # replace the found gaps with zeros in Hobs in order to pass subsequent
        # checks. Nans will be restored after swe calculation.
        Hobs = np.where(zeropadded_gap_idxs, 0., Hobs)

    if np.any(np.isnan(Hobs)) and interpolate_small_gaps:
            Hobs = fill_small_gaps(
                Hobs,
                dates,
                max_gap_length,
                interpolation_method)

    # check for (remaining) missing values.
    if np.any(np.isnan(Hobs)):
        _raise_nans_error_message(
            ignore_zeropadded_gaps,
            ignore_zerofollowed_gaps,
            interpolate_small_gaps,
            max_gap_length,
        )

    if not np.all(Hobs >= 0):
        raise ValueError("DeltaSNOW: snow depth data must be positive")

    if not np.all(np.isreal(Hobs)):
        raise ValueError("DeltaSNOW: snow depth data must be numeric")

    if Hobs[0] != 0:
        raise ValueError(("DeltaSNOW: snow depth observations must start "
                          "with 0 or the first non nan entry \nneeds to be "
                          "zero if you ignore zeropadded or zerofollowed gaps"))

    # start and stop indices of nonzero chunks.
    start_idxs, stop_idxs = get_nonzero_chunk_idxs(Hobs)

    # check for date continuity.
    continuous, resolution = continuous_timedeltas_in_nonzero_chunks(
        dates,
        start_idxs,
        stop_idxs)
    if not continuous:
        raise ValueError(("DeltaSNOW: date column must be strictly "
                          "regular within \nchunks of consecutive nonzeros"))

    swe_allocation = np.zeros(len(Hobs))

    swe = _deltasnow_on_nonzero_chunks(
        Hobs,
        swe_allocation,
        start_idxs,
        stop_idxs,
        rho_max,
        rho_null,
        c_ov,
        k_ov,
        k,
        tau,
        eta_null,
        resolution,
    )

    if ignore_zeropadded_gaps or ignore_zerofollowed_gaps:
        # restore nans in zeropadded gaps.
        swe = np.where(zeropadded_gap_idxs, np.nan, swe)

    # original R implementation (rewritten in ´.core´) returns SWE in ['mm']
    result = pd.Series(
        data=swe*0.001/UNIT_FACTOR[swe_output_unit],
        index=data.index,
        name=output_series_name,
    )

    return result
