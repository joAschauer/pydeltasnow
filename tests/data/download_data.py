"""
Download sample HS data from the sdbo database.

Intended to be run once and then the data is stored in files. If you want to
rerun this script, you need to be inside the WSL network and have the 'pysdbo'
package installed. You can get it from https://gitlab.wsl.ch/joAschauer/pysdbo
"""
import pandas as pd

import pysdbo

def download_from_database(station, start_year, end_year, drop_years):

    query = f"""select DATUM1D, STAT_ABK, HS
from v_beob
where date_to_yh(DATUM1D) in {tuple(range(start_year, end_year))}
and STAT_ABK in ('{station}')
order by DATUM1D
"""

    df = (pysdbo.query_as_dataframe(query)
          .rename(columns={'datum1d':'date', 'stat_abk':'station'})
          .assign(date=lambda x: pd.to_datetime(x['date']))
          .assign(hjahr=lambda x: x.date.dt.year.where(x.date.dt.month<8, x.date.dt.year+1))
          .assign(hs=lambda x: x['hs']/100)  # convert to [m]
          )

    df = (df
          .loc[~df['hjahr'].isin(drop_years), :]
          .reset_index(drop=True)
          )

    return df


def main():
    data_5wj = download_from_database(
        "5WJ",
        1980,
        2020,
        [1980,1981,1982,1990,1992]  # these years do not begin with zero.
    )
    data_5wj.to_csv("hs_data_5WJ.csv")

    data_5df = download_from_database(
        "5DF",
        1980,
        2019,
        [2001,2005]  # 2001: two missing entries, 2005: non regular dates
    )
    data_5df.to_csv("hs_data_5DF.csv")

    data_1ad = download_from_database(
         "1AD",
        1980,
        2000,
        []
    )
    data_1ad.to_csv("hs_data_1AD.csv")


if __name__ == '__main__':
    main()
