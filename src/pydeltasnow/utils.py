"""
This module contains utility functions for missing value handling and daterange
continuity validation.
"""

import pandas as pd
import numpy as np
from numba import njit

from pydeltasnow import __version__

__author__ = "Johannes Aschauer"
__copyright__ = "Johannes Aschauer"
__license__ = "GPL-2.0-or-later"

ONE_HOUR = np.timedelta64(1, 'h')

@njit
def continuous_timedeltas(dr):
    """
    Check for continuity on dates

    Parameters
    ----------
    dr : 1D :class:`numpy.ndarray` with :class:`numpy.datetime64`  dtype
        Input date range.
    Returns
    -------
    continuous : bool
        Whether the datetime array is evenly spaced.
    resolution : float
        The time resolution in hours.

    """
    if len(dr) <= 1:
        continuous = True
    else:
        # np.gradient not working in numba
        tdeltas = np.zeros(len(dr)-1, dtype='timedelta64[ns]')
        for i in range(len(dr)-1):
            tdeltas[i] = dr[i+1]-dr[i]

        # check if all deltas are equal
        continuous = np.all(tdeltas == tdeltas[0])
        # get time resolution in hours.
        # one hour timedelta has to be moved outside numba, see this issue for
        # reference: https://github.com/numba/numba/issues/1750
        resolution = tdeltas[0] / ONE_HOUR

    return continuous, resolution


@njit
def continuous_timedeltas_in_nonzero_chunks(
    dr,
    start_idxs,
    stop_idxs
):
    """
    Check that every non-zero HS chunk has continuous dates and same resolution.

    Parameters
    ----------
    dr : 1D :class:`numpy.ndarray` with :class:`numpy.datetime64`  dtype
        Input date range.
    start_idxs: 1D :class:`numpy.ndarray`
        Indices where a nonzero chunk begins.
    stop_idxs: 1D :class:`numpy.ndarray`
        Indices where a nonzero chunk ends.

    Returns
    -------
    continuous : bool
        Whether the chunks are evenly spaced with equal resolution.
    resolution : float
        The time resolution in hours.
    """

    cont = np.zeros(len(start_idxs), dtype='bool') # is chunk continuous
    res = np.zeros(len(start_idxs)) # chunk time resolutions
    for i, (start, stop) in enumerate(zip(start_idxs, stop_idxs)):
        cont[i], res[i] = continuous_timedeltas(dr[start:stop])

    continuous = np.all(np.array([np.all(cont), np.all(res==res[0])]))
    resolution = res[0]
    return continuous, resolution


@njit
def get_nonzero_chunk_idxs(Hobs):
    """
    Return start and stop indices of consecutive nonzero chunks in Hobs.

    A nonzero chunk is a slice in Hobs of values not zero. Nans are 
    treated as nonzeros. The slices which are retured from this function
    include one leading and one trailing zero since the main deltasnow 
    loop wants an input HS array to start and end with zero. Exceptions
    are when Hobs does not start and/or end with zero (then the first start
    and/or last stop index is not pointing to a zero).

    Note that when you slice `Hobs` with `Hobs[start:stop]` the stop value is 
    non-inclusive (https://stackoverflow.com/a/509295).

    Parameters
    ----------
    Hobs : 1D :class:`numpy.ndarray` of floats
        input HS data

    Returns
    -------
    start_idxs: 1D :class:`numpy.ndarray`
        Indices where a nonzero chunk begins.
    stop_idxs: 1D :class:`numpy.ndarray`
        Indices where a nonzero chunk ends.
    """

    start_idxs = []
    stop_idxs = []

    # if first value of Hobs is not zero, set 0 as first start_idx
    if Hobs[0] != 0:
        start_idxs.append(0)

    for i in range(len(Hobs)):
        if i < len(Hobs)-1:
            if Hobs[i] == 0. and Hobs[i+1]!=0:
                start_idxs.append(i)
        if i > 0:
            if Hobs[i] == 0. and Hobs[i-1]!=0:
                stop_idxs.append(i)

    # if last value not zero, set last idx of Hobs as last stop_idx
    if len(stop_idxs) < len(start_idxs):
        stop_idxs.append(len(Hobs))
    return np.array(start_idxs), np.array(stop_idxs)


@njit
def get_zeropadded_gap_idxs(
    Hobs,
    require_leading_zero,
):
    """
    Get indices of Nan data-gaps in Hobs that are surrounded or followed by
    zeros.

    Parameters
    ----------
    Hobs : 1D :class:`numpy.ndarray` of floats
        input HS data

    require_leading_zero : bool
        Whether to include gaps that do not have a leading zero but have a 
        trailing zero.

    Returns
    -------
    zeropadded_gap_idxs : 1D :class:`numpy.ndarray` of bools
    """
    zeropadded_gap_idxs = np.zeros(len(Hobs), dtype='bool')

    start_idxs = []
    stop_idxs = []

    gap = False
    start = len(Hobs)
    for i in range(len(Hobs)):
        if i == 0 and np.isnan(Hobs[i]):
            start = i
            gap = True

        if i > 0:
            if require_leading_zero:
                if np.isnan(Hobs[i]) and Hobs[i-1] == 0:
                    start = i
                    gap = True
            else:
                if np.isnan(Hobs[i]) and not np.isnan(Hobs[i-1]):
                    start = i
                    gap = True

        if i < len(Hobs)-1:
            if np.isnan(Hobs[i]) and Hobs[i+1] == 0:
                if gap:
                    start_idxs.append(start)
                    stop_idxs.append(i+1)

        if not np.isnan(Hobs[i]):
            gap=False

    # if last value also nan, set last idx of Hobs as last stop_idx
    if gap:
        if len(start_idxs)>0 and start != start_idxs[-1]:
            start_idxs.append(start)
            stop_idxs.append(len(Hobs))
        elif start < len(Hobs):
            start_idxs.append(start)
            stop_idxs.append(len(Hobs))

    if len(start_idxs)>0:
        for start_i, stop_i in zip(start_idxs, stop_idxs):
            zeropadded_gap_idxs[start_i:stop_i] = True

    return zeropadded_gap_idxs


@njit
def get_small_gap_idxs(
    Hobs,
    dates,
    max_gap_length,
):
    """
    Create a boolean mask for valid small gaps.

    Gap needs to be surrounded by values
    Dates need to be continuous between day before gap and day after gap

    Parameters
    ----------
    Hobs : 1D :class:`numpy.ndarray` of floats
        input HS data
    dates : 1D :class:`numpy.ndarray` of :class:`numpy.datetime64`  dtype
        timestamps of the snow depth observations.
    max_gap_length : int
        Only gaps shorter or equal max_gap_length are valid.


    Returns
    -------
    small_gap_idxs : 1D :class:`numpy.ndarray` of bools

    """
    small_gap_idxs = np.zeros(len(Hobs), dtype='bool')

    start_idxs = []
    stop_idxs = []

    gapl = 0  # counter of active gap length
    active_gap = False
    valid_gap = False

    start = len(Hobs)
    for i in range(len(Hobs)):
        if active_gap:
            gapl = gapl+1

        if gapl > max_gap_length:
            valid_gap = False

        if i > 0:
            if not np.isnan(Hobs[i-1]) and np.isnan(Hobs[i]):
                start = i
                valid_gap=True
                active_gap = True

        if i < len(Hobs):
            if np.isnan(Hobs[i-1]) and not np.isnan(Hobs[i]):
                if valid_gap and continuous_timedeltas(dates[start-1:i+1])[0]:
                    start_idxs.append(start)
                    stop_idxs.append(i)

                gapl = 0
                active_gap = False
                valid_gap=False


    if len(start_idxs)>0:
        for start_i, stop_i in zip(start_idxs, stop_idxs):
            small_gap_idxs[start_i:stop_i] = True

    return small_gap_idxs


def fill_small_gaps(
    Hobs,
    dates,
    max_gap_length,
    method='linear'
):
    """
    Interpolate small gaps in Hobs.

    No extrapolation: data points before and after gap needed.
    Date continuity in the filled gaps + leading and trailing data point is
    ensured.

    Parameters
    ----------
    Hobs : 1D :class:`numpy.ndarray`
        Snow depth data.
    dates : 1D :class:`numpy.ndarray` of :class:`numpy.datetime64`  dtype
        timestamps of the snow depth observations.
    max_gap_length : int
        Only gaps shorter or equal max_gap_length are interpolated.
    method : str, optional
        Interpolation method which will be passed to 
        :func:`pandas.Series.interpolate`. The default is 'linear'.

    Returns
    -------
    Hobs_intepolated : 1D :class:`numpy.ndarray` 
        Snow depth data with filled gaps.

    """
    valid_gap_mask = get_small_gap_idxs(Hobs, dates, max_gap_length)
    interpolated = pd.Series(Hobs).interpolate(method=method).to_numpy()
    Hobs_interpolated = np.where(valid_gap_mask, interpolated, Hobs)
    return Hobs_interpolated
