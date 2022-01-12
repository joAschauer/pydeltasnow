# -*- coding: utf-8 -*-
"""
Reimplementation of the delta.snow model by Winkler et al 2021:
    
Winkler, M., Schellander, H., and Gruber, S.: Snow water equivalents
exclusively from snow depths and their temporal changes: the delta.snow model,
Hydrol. Earth Syst. Sci., 25, 1165-1187, doi: 10.5194/hess-25-1165-2021, 2021. 

The core of this code is mainly based on the work of Manuel Theurl:
https://bitbucket.org/atraxoo/snow_to_swe/src/master/

This version uses numba just-in-time compilation for significant performance
improvements.

Created on Mon Nov 29 10:30:03 2021

@author: Johannes Aschauer (johannes.aschauer[äääht]slf.ch)
"""
import pandas as pd
import numpy as np
from numba import njit

from .core import deltasnow_snowpack_evolution

from .utils import (
    continuous_timedeltas,
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
    Hobs : 1D np.array of floats 
        Measured snow height. Needs to be in [m].
    swe_out : 1D np.array of floats
        preallocated swe array where the output is stored to. Same shape as Hobs.
    start_idxs : np.array of int
        Start indices of windows with nonzeros in Hobs.
    stop_idxs : np.array of int
        Last indices of windows with nonzeroes in Hobs.
    rho_max : float, optional
        Maximum density of an individual snow layer produced by the deltasnow 
        model in [kg/m3], rho_max needs to be positive. The default is 401.2588.
    rho_null : float, optional
        Fresh snow density for a newly created layer [kg/m3], rho_null needs to
        be positive. The default is 81.19417.
    c_ov : float, optional
        Overburden factor due to fresh snow [-], c_ov needs to be positive. The
        default is 0.0005104722.
    k_ov : float, optional
        Defines the impact of the individual layer density on the compaction due
        to overburden [-], k_ov need to be in the range [0,1].
        The default is 0.37856737.
    k : float, optional
        Exponent of the exponential-law compaction [m3/kg], k needs to be
        positive. The default is 0.02993175.
    tau : float, optional
        Uncertainty bound [m], tau needs to be positive.
        The default is 0.02362476.
    eta_null : float, optional
        Effective compactive viscosity of snow for "zero-density" [Pa s].
        The default is 8523356.
    resolution : float
        Timedelta in hours between snow observations.

    Returns
    -------
    swe_out : np.array
    
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


def swe_delta_snow(
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
    ignore_trailingzero_gaps=False,
    interpolate_small_gaps=False,
    max_gap_length=3,
    interpolation_method='linear',
):
    """
    Calculate snow water equivalent (SWE) with the delta.snow model on a snow 
    depth (HS) timeseries.
    
    Differences to the original R implementation of Winkler et al 2021:
        - Accepts a pd.DataFrame with columns 'date' and 'hs' or pd.Series with
          pd.DatetimeIndex and HS data.
        - The time resolution (timestep in R implementation) will be automatically 
          sniffed from the 'date' column or DatetimeIndex
        - The user can specify the input and output units of the HS and SWE
          measurement series, respectively.
        - The model can accept breaks in the date series when these are small
          or when a break is sourrounded by zeros or is followed by a zero.
          This can be useful for measurement series that are not continued in 
          summer or which suddenly end e.g. by the end of April.
          Accordingly, the user can specify how to deal with missing values in
          a measurement series. There are three parameters that control NaN
          handling:
            - 'ignore_zeropadded_gaps'
            - 'ignore_trailingzero_gaps'
            - 'interpolate_small_gaps'
          Note that the runtime efficiency of the model will decrease when one 
          or several of these options are turnded on.
        - A pd.Series with the dates as pd.DatetimeIndex is returned.


    Parameters
    ----------
    data : pd.DataFrame
        Either a
            - pd.DataFrame with columns 'hs and 'date'
            - pd.Series with pd.DatetimeIndex
    rho_max : float, optional
        Maximum density of an individual snow layer produced by the deltasnow 
        model in [kg/m3], rho_max needs to be positive. The default is 401.2588.
    rho_null : float, optional
        Fresh snow density for a newly created layer [kg/m3], rho_null needs to
        be positive. The default is 81.19417.
    c_ov : float, optional
        Overburden factor due to fresh snow [-], c_ov needs to be positive. The
        default is 0.0005104722.
    k_ov : float, optional
        Defines the impact of the individual layer density on the compaction due
        to overburden [-], k_ov need to be in the range [0,1].
        The default is 0.37856737.
    k : float, optional
        Exponent of the exponential-law compaction [m3/kg], k needs to be
        positive. The default is 0.02993175.
    tau : float, optional
        Uncertainty bound [m], tau needs to be positive.
        The default is 0.02362476.
    eta_null : float, optional
        Effective compactive viscosity of snow for "zero-density" [Pa s].
        The default is 8523356.
    hs_input_unit : str in {'mm', 'cm', 'm'}
        The unit of the input snow depth. The default is 'm'.
    swe_output_unit : str in {'mm', 'cm', 'm'}
        The unit of the output snow water equivalent. The default is 'mm'.
    ignore_zeropadded_gaps : bool
        Whether to ignore gaps that have leading and trailing zeros. The 
        resulting SWE series will contain NaNs at the same positions. These 
        gaps are also ignored when you use `ignore_trailingzero_gaps`.
    ignore_trailingzero_gaps : bool
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
        pandas.Series.interpolate(). See the documentation for valid options.
        The default is 'linear'.
    

    Raises
    ------
    ValueError
        If any of the constraints on the data is violated.

    Returns
    -------
    swe : pd.Series 
        Calculated SWE with 'date' column of the input data as pd.DatetimeIndex.

    """
    data = data.copy()
    
    assert hs_input_unit in UNIT_FACTOR.keys(), "swe.deltasnow: hs_input_unit has to be in {'mm', 'cm', 'm'}"
    assert swe_output_unit in UNIT_FACTOR.keys(), "swe.deltasnow: swe_output_unit has to be in {'mm', 'cm', 'm'}"
    
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.Series):
            raise ValueError("swe.deltasnow: data must be pd.DataFrame or pd.Series")
        else:
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("swe.deltasnow: pd.Series needs pd.DatetimeIndex as index.")
            
            else: # convert series to dataframe
                data = (data
                        .rename('hs')
                        .reset_index(drop=False)
                        .rename(columns={'index': 'date',
                                         data.index.name: 'date'},
                                errors='ignore')
                        )
    
    if any([c not in data.columns for c in ['date', 'hs']]):
        raise ValueError(("swe.deltasnow: data must be a pd.Series with pd.DatetimeIndex or a pd.Dataframe" 
                          "containing at least two columns named 'hs' and 'date'"))
    
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')
    Hobs = data['hs'].mul(UNIT_FACTOR[hs_input_unit]).to_numpy()
    
    if ignore_zeropadded_gaps or ignore_trailingzero_gaps:
        if ignore_trailingzero_gaps:
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
                data['date'].to_numpy(),
                max_gap_length,
                interpolation_method)

    # check for (remaining) missing values.
    has_nans = np.any(np.isnan(Hobs))
    if (has_nans
            and not ignore_zeropadded_gaps
            and not interpolate_small_gaps):
        raise ValueError("swe.deltasnow: snow depth data must not be NaN.")
    elif (has_nans
            and ignore_zeropadded_gaps
            and not interpolate_small_gaps):
        raise ValueError(("swe.deltasnow: your data contains NaNs surrounded "
                          "by non-zeros."))
    elif (has_nans
            and ignore_zeropadded_gaps
            and interpolate_small_gaps):
        raise ValueError(("swe.deltasnow: your data contains gaps at the end "
                          "or beginning of your \nseries or gaps longer than "
                          f"{max_gap_length} timesteps"))

    if not np.all(Hobs >= 0):
        raise ValueError("swe.deltasnow: snow depth data must be positive")

    if not np.all(np.isreal(Hobs)):
        raise ValueError("swe.deltasnow: snow depth data must be numeric")

    if Hobs[0] != 0:
        raise ValueError(("swe.deltasnow: snow depth observations must start "
                          "with 0 or the first non nan entry \nneeds to be "
                          "zero if you ignore zeropadded gaps"))

    # start and stop indices of nonzero chunks.
    start_idxs, stop_idxs = get_nonzero_chunk_idxs(Hobs)
    
    # check for date continuity.
    continuous, resolution = continuous_timedeltas(data['date'].to_numpy())
    if not continuous:
        continuous, resolution = continuous_timedeltas_in_nonzero_chunks(
            data['date'].to_numpy(),
            start_idxs,
            stop_idxs)
        if not continuous:
            raise ValueError(("swe.deltasnow: date column must be strictly "
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
    
    if ignore_zeropadded_gaps or ignore_trailingzero_gaps:
        # restore nans in zeropadded gaps.
        swe = np.where(zeropadded_gap_idxs, np.nan, swe)

    # original R implementation (rewritten in ´.core´) returns SWE in ['mm']
    result = pd.Series(
        data=swe*0.001/UNIT_FACTOR[swe_output_unit],
        index=data['date'],
        name='swe_deltasnow',
    )
    
    return result
