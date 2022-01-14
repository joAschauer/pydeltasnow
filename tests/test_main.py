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

from pydeltasnow import swe_delta_snow

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
def test_swe_delta_snow_against_nixmass(
    input_hs_data,
    nixmass_swe_data,
    request
):
    input_hs_data = request.getfixturevalue(input_hs_data)
    nixmass_swe_data = request.getfixturevalue(nixmass_swe_data)
    swe_pydeltasnow = swe_delta_snow(input_hs_data)
    pd.testing.assert_series_equal(swe_pydeltasnow, nixmass_swe_data)
