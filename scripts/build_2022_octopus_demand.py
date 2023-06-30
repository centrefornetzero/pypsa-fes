# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023 @LukasFrankenQ, The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
This script obtains 2022 electric load data for a subset of European countries - 
the ones used in studies conducted at Centre for Net Zero.

Mainland Europe data is taken from ENTSO-E API according to the github
on the relevant Python based API <https://github.com/EnergieID/entsoe-py>`_.

To run this, an API key needs to be obtained from the website. It should 
be stored in a textfile noted in snakemake.config["entsoe_key_file"]


The data for GB is obtained from ESO, and is the national day-ahead demand forecast
https://data.nationalgrideso.com/demand/1-day-ahead-demand-forecast

The timeseries is then scaled to match total GB electricity demand according to
statista:
https://www.statista.com/statistics/323381/total-demand-for-electricity-in-the-united-kingdom-uk/
321.29 TWh


Relevant Settings
-----------------

.. code:: yaml
    
    snapshots:


.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`entsoe_key_file`

Inputs
------

[1] GB demand data taken from
    https://data.nationalgrideso.com/demand/historic-demand-data
    2022 data, column ND

Outputs
-------

- ``resources/load.csv``:
"""


import logging

logger = logging.getLogger(__name__)

import sys
import dateutil
import numpy as np
import pandas as pd
from _helpers import configure_logging
from pandas import Timedelta as Delta

from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from requests.exceptions import HTTPError, ConnectTimeout


"""
    
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_load_series")
    
    configure_logging(snakemake)

    countries = set(snakemake.config["countries"])
    countries.remove("GB")
    countries = list(countries)
    
    snapshots = pd.date_range(freq="h", **snakemake.config["snapshots"])
    years = slice(snapshots[0], snapshots[-1])

    interpolate_limit = snakemake.config["load"]["interpolate_limit"]
    replace_thresh = snakemake.config["load"]["time_shift_for_large_gaps"]

    # length in timesteps as in dataframe
    replace_thresh = len(pd.date_range(
        pd.Timestamp("2022"),
        pd.Timestamp("2022") + pd.Timedelta(replace_thresh),
        freq=pd.infer_freq(snapshots)))

    with open('entsoe_api_key.txt', 'r') as f:
        key = f.readlines()[0]

    client = EntsoePandasClient(api_key=key)

    entsoe_start = (snapshots[0] - pd.Timedelta(1, unit="day")).tz_localize("Europe/Brussels")
    entsoe_end = (snapshots[-1] + pd.Timedelta(1, unit="day")).tz_localize("Europe/Brussels")

    start = snapshots[0]
    end = snapshots[-1]

    # index_load = client.query_load("DE",
    #                # start=entsoe_start,
    #                     end=entsoe_end)
    # index = index_load.resample("h").mean().loc[start:end].index.values
    index = pd.date_range(freq='h', **snakemake.config["snapshots"])

    load_df = pd.DataFrame(index=index)
    backup_load = pd.read_csv(snakemake.input["default_load"], parse_dates=True, index_col=0)
    backup_load.index = index

    for country in countries:

        try:
            load = client.query_load(country,
                            start=entsoe_start,
                            end=entsoe_end)
        except HTTPError:
            print(f"Unable to connect to ENTSO-E, replacing with backup for {country}")
            load_df[country] = backup_load[country]
            continue
        except ConnectTimeout:
            print(f"Unable to connect to ENTSO-E, replacing with backup for {country}")
            load_df[country] = backup_load[country]
            continue

        load = load.resample("h").mean().loc[start:end]
        try:
            load.index = index
        except ValueError:
            print(f"Unsufficient data, replacing with backup load for {country}")
            load_df[country] = backup_load[country]
            continue

        load = load.interpolate(method="linear", limit=interpolate_limit)

        if load.isna().sum().sum() >= replace_thresh:
            load_df[country] = backup_load[country]
        else:
            load_df[country] = load

    # get GB demand
    gb_load = pd.read_csv(snakemake.input["gb_demand"], parse_dates=True)
    gb_load["datetime"] = gb_load.apply(
        lambda row: pd.Timestamp(row.SETTLEMENT_DATE) + pd.Timedelta(30*(row.SETTLEMENT_PERIOD-1), unit="min"),
        axis=1
    )
    gb_load.index = gb_load["datetime"]
    gb_load = gb_load["TSD"].resample("h").mean().interpolate(method="linear")

    gb_load = gb_load / gb_load.sum() * (321.29 * 1e6)

    load_df.index.tz_localize("Europe/Brussels")

    load_df["GB"] = gb_load

    assert not load.isna().any().any(), (
        "Load data contains nans. Adjust the data."
    )

    load_df.to_csv(snakemake.output["load"])
   
"""

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_load_series")
    
    configure_logging(snakemake)

    logger.info("Transferring demand data from 2013 to 2022 for mainland Europe.")
    logger.warning("Not yet implemented dynamic ENTSO-E download pipeline for mainland EU")

    load = pd.read_csv(snakemake.input["default_load"], parse_dates=True, index_col=0)
    load.index = pd.date_range("2022", "2022-12-31 23:00", freq="h")

    # [1]
    esoload = pd.read_csv(snakemake.input["gb_demand"], parse_dates=True)

    esoload["dt"] = esoload.apply(lambda row: pd.Timestamp(row.SETTLEMENT_DATE) + 
                            pd.Timedelta(30*(row.SETTLEMENT_PERIOD-1), unit="min"),
                            axis=1)

    esoload.index = esoload["dt"]
    # esoload.index = esoload.index.tz_localize("UTC")
    esoload = esoload.resample("h").mean()
    esoload = esoload["ND"]

    load["GB"] = esoload.values
    load["GB"] = load["GB"].interpolate(method="linear") # we know there is a single nan in this dataset

    assert not load.isna().any().any(), (
        "Load data contains nans. Adjust the data."
    )

    load.to_csv(snakemake.output["load"])