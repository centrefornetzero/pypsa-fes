# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023 Lukas Franken; inspired by the PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8

"""
Change the p_set timeseries in a prepared network to the values
observed in flexibility events
"""

import logging
import re
import pypsa

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)


def build_flex_network(n, td_file, event_length=pd.Timedelta(30, unit="min")):
    """
    Changes GB time series for p_set according to turndown event
    
    """

    td = pd.read_csv(td_file, index_col=0, parse_dates=True)

    logger.info("received turndown")
    logger.info(td)
    
    model_year = n.snapshots[0].year
    
    if model_year != td.index[0].year:
        logger.warning("Currently setting turndown event year to the models' year.")
        td.index = td.reset_index().iloc[:,0].apply(lambda x: x.replace(year=model_year))

    gb_buses = [col for col in n.loads_t.p_set.columns if "GB0" in col]
    base = n.loads_t.p_set[gb_buses]
    
    if pd.infer_freq(td.index) is None:

        logger.warning("Assumes unit of td is kWh")

        relative_mag = ((td * 1e-3).values / base.loc[td.index].sum(axis=1).values).mean()
        relative_mag = np.around(relative_mag*100, decimals=3)
        relative_mag_std = ((td * 1e-3).values / base.loc[td.index].sum(axis=1).values).std()
        relative_mag_std = np.around(relative_mag_std*100, decimals=3)
        logger.info(f"On average {relative_mag}+-{relative_mag_std} % of total demand was turned down.")
        
        factor = event_length.seconds / pd.Timedelta(1, unit="hour").seconds
        turndown_shares = n.loads_t.p_set[gb_buses].sum() / n.loads_t.p_set[gb_buses].sum().sum()

        base.loc[td.index] = base.loc[td.index].values - turndown_shares.values * factor * td.values * 1e-3

        n.loads_t.p_set[gb_buses] = base
    else:
        logging.error("Format of turndown data currently not supported; load unchanged!")

    n.export_to_netcdf(snakemake.output["network"])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_flex_network")
    
    n = pypsa.Network(snakemake.input["network"])
    
    n = build_flex_network(n, snakemake.input["flex_data"]) 
