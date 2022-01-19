import pytest
import numpy as np
import pandas as pd

from pydeltasnow.utils import (
    continuous_timedeltas,
    get_nonzero_chunk_idxs,
    get_small_gap_idxs,
    get_zeropadded_gap_idxs,
    )

__author__ = "Johannes Aschauer"
__copyright__ = "Johannes Aschauer"
__license__ = "GPL-2.0-or-later"


@pytest.fixture
def dates_continuous_one_day():
    dates = pd.date_range(start='2000-01-01', periods=9, freq='D').to_numpy()
    return dates


@pytest.fixture
def dates_incontinuous_one_day():
    dates = pd.date_range(start='2000-01-01', periods=9, freq='D').to_numpy()
    dates[4:] = dates[4:] + np.timedelta64(2, "D")
    return dates


@pytest.fixture
def dates_continuous_two_days():
    dates = pd.date_range(start='2000-01-01', periods=9, freq='2D').to_numpy()
    return dates


@pytest.mark.parametrize(
    "input_dates, continuous_expected, resolution_expected",
    [
        ("dates_continuous_one_day", True, 24),
        ("dates_incontinuous_one_day", False, 24),
        ("dates_continuous_two_days", True, 48),
    ],
)
def test_continuous_timedeltas(
    input_dates, 
    continuous_expected,
    resolution_expected,
    request
):
    input_dates = request.getfixturevalue(input_dates)
    contiuous, resolution = continuous_timedeltas(input_dates)
    assert contiuous == continuous_expected
    assert resolution == resolution_expected


@pytest.mark.parametrize(
    "sample, start_expected, stop_expected",
    [
        (np.array([0,0,1,1,1,1,0,0]), np.array([1]), np.array([6])),
        (np.array([1,1,1,0,0]), np.array([0]), np.array([3])),
        (np.array([0,0,0,1,1,1]), np.array([2]), np.array([6])),
        (np.array([0,1,0,3,4,0,6,0,0,9,0]), np.array([0,2,5,8]), np.array([2,5,7,10])),
        (np.zeros(10), np.array([]), np.array([])),
    ],
)
def test_get_nonzero_chunk_idxs(sample, start_expected, stop_expected):
    start_out, stop_out = get_nonzero_chunk_idxs(sample)
    np.testing.assert_array_equal(start_out, start_expected)
    np.testing.assert_array_equal(stop_out, stop_expected)


def test_get_zeropadded_gap_idxs():
    # nans at beginning of series
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,np.nan,0]), True),
        np.array([True, True, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,np.nan,1]), True),
        np.array([False, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,0,1]), True),
        np.array([True, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,1,1]), True),
        np.array([False, False, False])
        )

    # end of series
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([0,np.nan,np.nan]), True),
        np.array([False,True, True])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,np.nan,np.nan]), True),
        np.array([False, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,0,np.nan]), True),
        np.array([False,False, True])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,1,np.nan]), True),
        np.array([False, False, False])
        )

    #single_gap
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,0,np.nan,0,1]), True),
        np.array([False, False, True, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,1,np.nan,0,1]), True),
        np.array([False, False, False, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,0,np.nan,1,1]), True),
        np.array([False, False, False, False, False])
        )

    # long series:
    long_in = np.array([np.nan,np.nan,0,3,4,5,6,7,7,3,np.nan,np.nan,0,9,0,np.nan,np.nan,np.nan,0,7,0,np.nan,np.nan,8,0,0,np.nan,np.nan,np.nan], dtype='float64')
    long_expected = np.array([ True,  True, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True,  True,  True,
       False, False, False, False, False, False, False, False,  True,
        True,  True])
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(long_in, True),
        long_expected)

    # gaps with trailing zero only:
    # =============================
    # nans at beginning of series
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,np.nan,0]), False),
        np.array([True, True, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,np.nan,1]), False),
        np.array([False, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,0,1]), False),
        np.array([True, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([np.nan,1,1]), False),
        np.array([False, False, False])
        )

    # end of series
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([0,np.nan,np.nan]), False),
        np.array([False,True, True])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,np.nan,np.nan]), False),
        np.array([False, True, True])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,0,np.nan]), False),
        np.array([False,False, True])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,1,np.nan]), False),
        np.array([False, False, True])
        )

    #single_gap
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,0,np.nan,0,1]), False),
        np.array([False, False, True, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,1,np.nan,0,1]), False),
        np.array([False, False, True, False, False])
        )
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(np.array([1,0,np.nan,1,1]), False),
        np.array([False, False, False, False, False])
        )

    # long series:
    long_in = np.array([np.nan,np.nan,0,3,4,5,6,7,7,3,np.nan,np.nan,0,9,0,np.nan,np.nan,np.nan,0,7,0,np.nan,np.nan,8,0,0,np.nan,np.nan,np.nan], dtype='float64')
    long_expected = np.array([ True,  True, False, False, False, False, False, False, False,
       False, True, True, False, False, False,  True,  True,  True,
       False, False, False, False, False, False, False, False,  True,
        True,  True])
    np.testing.assert_array_equal(
        get_zeropadded_gap_idxs(long_in, False),
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
