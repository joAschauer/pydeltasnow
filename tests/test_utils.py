import pytest
import numpy as np
import pandas as pd

from pydeltasnow.utils import (
    get_small_gap_idxs,
    get_zeropadded_gap_idxs,
    )

__author__ = "Johannes Aschauer"
__copyright__ = "Johannes Aschauer"
__license__ = "GPL-2.0-or-later"


def test_continuous_timedeltas():
    # TODO
    pass


def test_get_nonzero_chunk_idxs():
    # TODO
    # np.testing.assert_array_equal(
    #     get_nonyero_chunk_idxs(np.array([np.nan,np.nan,0])),
    #     np.array([True, True, False])
    #     )
    pass


def test_get_zeropadded_gap_idxs():
    # nans at beginning of series
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,np.nan,0])),
        np.array([True, True, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,np.nan,1])),
        np.array([False, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,0,1])),
        np.array([True, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,1,1])),
        np.array([False, False, False])
        )

    # end of series
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([0,np.nan,np.nan])),
        np.array([False,True, True])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,np.nan,np.nan])),
        np.array([False, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,0,np.nan])),
        np.array([False,False, True])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,1,np.nan])),
        np.array([False, False, False])
        )

    #single_gap
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,0,np.nan,0,1])),
        np.array([False, False, True, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,1,np.nan,0,1])),
        np.array([False, False, False, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,0,np.nan,1,1])),
        np.array([False, False, False, False, False])
        )

    # long series:
    long_in = np.array([np.nan,np.nan,0,3,4,5,6,7,7,3,np.nan,np.nan,0,9,0,np.nan,np.nan,np.nan,0,7,0,np.nan,np.nan,8,0,0,np.nan,np.nan,np.nan], dtype='float64')
    long_expected = np.array([ True,  True, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True,  True,  True,
       False, False, False, False, False, False, False, False,  True,
        True,  True])
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(long_in),
        long_expected)


def test_get_small_gap_idxs():
    # 3 nans in the middle, continuous date series:
    hs_in = np.array([1,1,1,np.nan, np.nan, np.nan, 1, 1, 1])
    dates_in = pd.date_range(start='2000-01-01', periods=9, freq='D').to_numpy()
    assert(len(dates_in)==len(hs_in))
    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 4),
        np.array([False,False,False,True,True,True,False,False,False])
        )

    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 3),
        np.array([False,False,False,True,True,True,False,False,False])
        )

    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 2),
        np.array([False,False,False,False,False,False,False,False,False])
        )

    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 1),
        np.array([False,False,False,False,False,False,False,False,False])
        )

    # 3 nans in the middle, incontinuous date series:
    # wrong date in the gap
    hs_in = np.array([1,1,1,np.nan, np.nan, np.nan, 1, 1, 1])
    dates_in = pd.date_range(start='2000-01-01', periods=9, freq='D').to_numpy()
    dates_in[4] = pd.Timestamp(2000, 2, 1).to_numpy()
    assert(len(dates_in)==len(hs_in))
    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 3),
        np.array([False,False,False,False,False,False,False,False,False])
        )

    # wrong date at the the first entry after gap
    hs_in = np.array([1,1,1,np.nan, np.nan, np.nan, 1, 1, 1])
    dates_in = pd.date_range(start='2000-01-01', periods=9, freq='D').to_numpy()
    dates_in[6] = pd.Timestamp(2000, 2, 1).to_numpy()

    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 3),
        np.array([False,False,False,False,False,False,False,False,False])
        )

    # wrong date at the the last entry before gap
    hs_in = np.array([1,1,1,np.nan, np.nan, np.nan, 1, 1, 1])
    dates_in = pd.date_range(start='2000-01-01', periods=9, freq='D').to_numpy()
    dates_in[2] = pd.Timestamp(2000, 2, 1).to_numpy()
    assert(len(dates_in)==len(hs_in))
    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 3),
        np.array([False,False,False,False,False,False,False,False,False])
        )

    # wrong date at the very last entry:
    hs_in = np.array([1,1,1,np.nan, np.nan, np.nan, 1])
    dates_in = pd.date_range(start='2000-01-01', periods=7, freq='D').to_numpy()
    dates_in[-1] = pd.Timestamp(2000, 2, 1).to_numpy()
    assert(len(dates_in)==len(hs_in))
    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 3),
        np.array([False,False,False,False,False,False,False])
        )

    # wrong date at the very first entry:
    hs_in = np.array([1,np.nan, np.nan, np.nan, 1])
    dates_in = pd.date_range(start='2000-01-01', periods=5, freq='D').to_numpy()
    dates_in[0] = pd.Timestamp(2000, 2, 1).to_numpy()
    assert(len(dates_in)==len(hs_in))

    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 2),
        np.array([False,False,False,False,False])
        )

    # gap at the beginning:
    hs_in = np.array([np.nan, np.nan, np.nan, 1])
    dates_in = pd.date_range(start='2000-01-01', periods=4, freq='D').to_numpy()
    assert(len(dates_in)==len(hs_in))

    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 5),
        np.array([False,False,False,False])
        )

     # gap at the end:
    hs_in = np.array([1,np.nan, np.nan, np.nan])
    dates_in = pd.date_range(start='2000-01-01', periods=4, freq='D').to_numpy()
    assert(len(dates_in)==len(hs_in))
    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 5),
        np.array([False,False,False,False])
        )

    # all nans
    hs_in = np.array([np.nan,np.nan, np.nan, np.nan])
    dates_in = pd.date_range(start='2000-01-01', periods=4, freq='D').to_numpy()
    assert(len(dates_in)==len(hs_in))
    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 5),
        np.array([False,False,False,False])
        )

    # no gaps
    hs_in = np.array([1,1,1,1])
    dates_in = pd.date_range(start='2000-01-01', periods=4, freq='D').to_numpy()
    assert(len(dates_in)==len(hs_in))
    np.testing.assert_array_equal(
        get_small_gap_idxs(hs_in, dates_in, 5),
        np.array([False,False,False,False])
        )
