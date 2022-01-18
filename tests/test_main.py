"""
Compare output of the pydeltasnow python implementation against the original
R implementation of the 'nixmass' package.

Notes on the test data and processing scheme. Since I did not want to make R
nor rpy2 a test dependency of this package, I decided to run the calculation
of the R model externally and provide the output as csv file in the
`tests/data` directory. This directory holds input HS data and SWE data
calculated with the nixmass R package respectively as well as scripts to get
this data. The fixtures in this module load the HS and SWE data.
"""
from distutils import dir_util
import os
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from pydeltasnow import swe_deltasnow

__author__ = "Johannes Aschauer"
__copyright__ = "Johannes Aschauer"
__license__ = "GPL-2.0-or-later"


@pytest.fixture
def datadir(tmpdir, request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    Adapted from: https://stackoverflow.com/a/29631801
    '''
    filename =  Path(request.module.__file__)
    test_dir = filename.parent / "data"

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


@pytest.fixture
def hs_5wj_as_df(datadir):
    df = (pd.read_csv(datadir.join("hs_data_5WJ.csv"))
          .loc[:, ["date", "hs"]]
          .assign(date=lambda x: pd.to_datetime(x['date']))
          )
    return df


@pytest.fixture
def hs_5df_as_df(datadir):
    df = (pd.read_csv(datadir.join("hs_data_5DF.csv"))
          .loc[:, ["date", "hs"]]
          .assign(date=lambda x: pd.to_datetime(x['date']))
          )
    return df


@pytest.fixture
def hs_1ad_as_df(datadir):
    df = (pd.read_csv(datadir.join("hs_data_1AD.csv"))
          .loc[:, ["date", "hs"]]
          .assign(date=lambda x: pd.to_datetime(x['date']))
          )
    return df


@pytest.fixture
def swe_5wj_as_series(datadir):
    return pd.read_csv(datadir.join("swe_data_5WJ.csv"),
                       parse_dates=['date'],
                       index_col='date').squeeze()


@pytest.fixture
def swe_5df_as_series(datadir):
    return pd.read_csv(datadir.join("swe_data_5DF.csv"),
                       parse_dates=['date'],
                       index_col='date').squeeze()


@pytest.fixture
def swe_1ad_as_series(datadir):
    return pd.read_csv(datadir.join("swe_data_1AD.csv"),
                       parse_dates=['date'],
                       index_col='date').squeeze()


@pytest.mark.parametrize(
    "input_hs_data, nixmass_swe_data",
    [
        ("hs_5wj_as_df", "swe_5wj_as_series"),
        ("hs_5df_as_df", "swe_5df_as_series"),
        ("hs_1ad_as_df", "swe_1ad_as_series")
    ],
)
def test_swe_deltasnow_against_nixmass(
    input_hs_data,
    nixmass_swe_data,
    request
):
    input_hs_data = request.getfixturevalue(input_hs_data)
    nixmass_swe_data = request.getfixturevalue(nixmass_swe_data)
    swe_pydeltasnow = swe_deltasnow(input_hs_data)
    pd.testing.assert_series_equal(swe_pydeltasnow, nixmass_swe_data)


@pytest.fixture
def hs_5wj_with_zeropadded_gaps(hs_5wj_as_df):
    df = hs_5wj_as_df
    hs = df['hs'].copy()
    hs.iloc[374:398] = np.nan
    hs.iloc[1438:1453] = np.nan
    df['hs'] = hs
    return df

@pytest.fixture
def swe_5wj_with_zeropadded_gaps(swe_5wj_as_series):
    s = swe_5wj_as_series.copy()
    s.iloc[374:398] = np.nan
    s.iloc[1438:1453] = np.nan
    return s

@pytest.fixture
def hs_5wj_with_zerofollowed_gaps(hs_5wj_as_df):
    df = hs_5wj_as_df
    hs = df['hs'].copy()
    hs.iloc[373:398] = np.nan
    hs.iloc[1079:1094] = np.nan
    df['hs'] = hs
    return df

@pytest.fixture
def swe_5wj_with_zerofollowed_gaps(swe_5wj_as_series):
    s = swe_5wj_as_series.copy()
    s.iloc[373:398] = np.nan
    s.iloc[1079:1094] = np.nan
    return s

def test_zeropadded_gaps(
    hs_5wj_with_zeropadded_gaps,
    swe_5wj_with_zeropadded_gaps
):
    swe_pydeltasnow = swe_deltasnow(hs_5wj_with_zeropadded_gaps,
                                     ignore_zeropadded_gaps=True)
    pd.testing.assert_series_equal(swe_pydeltasnow, swe_5wj_with_zeropadded_gaps)

    swe_pydeltasnow = swe_deltasnow(hs_5wj_with_zeropadded_gaps,
                                     ignore_zerofollowed_gaps=True)
    pd.testing.assert_series_equal(swe_pydeltasnow, swe_5wj_with_zeropadded_gaps)


def test_zerofollowed_gaps(
    hs_5wj_with_zerofollowed_gaps,
    swe_5wj_with_zerofollowed_gaps):

    with pytest.raises(ValueError):
        swe_pydeltasnow = swe_deltasnow(hs_5wj_with_zerofollowed_gaps,
                                         ignore_zeropadded_gaps=True)

    swe_pydeltasnow = swe_deltasnow(hs_5wj_with_zerofollowed_gaps,
                                     ignore_zerofollowed_gaps=True)
    pd.testing.assert_series_equal(swe_pydeltasnow, swe_5wj_with_zerofollowed_gaps)


def test_series_input(hs_5wj_as_df, swe_5wj_as_series):
    hs_series = hs_5wj_as_df.set_index('date', drop=True)['hs']
    hs_series.index = hs_series.index.rename("nonsense_index_name")
    swe_pydeltasnow = swe_deltasnow(hs_series)
    pd.testing.assert_series_equal(swe_pydeltasnow, swe_5wj_as_series)