# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023 Lukas Franken, The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Implements electricity demand time series for each regions of the
network, based on the the ESPENI dataset, see

'Calculating Great Britain`s half-hourly electrical demand from publicly
available data' by Wilson et al. 2021 

This dataset estimates true electricity demand data in the UK based on the
transmission level generation data from Elexon, added to electricity imports,
and added to estimated distribution level generation (as made availabel by
national grid ESO).

The timeseries is distributed among network regions based on total demand
estimations made in Future Energy Scenarios.

The scripts first retrieves the data from zenodo, then processes it.

It further assumes that the model includes embedded (distribution level)
generation capacities

Relevant Settings
-----------------   

.. code:: yaml

    snapshots:

Outputs
-------

- ``RESOURCES + electricity_demand_gb_s{simpl}_eso.csv``
"""

import logging

logger = logging.getLogger(__name__)

import pandas as pd

from _helpers import progress_retrieve, configure_logging


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_databundle")
        rootpath = ".."
    else:
        rootpath = "."
    configure_logging(
        snakemake
    )  # TODO Make logging compatible with progressbar (see PR #102)

    espeni_url = "https://zenodo.org/record/3884859/files/espeni.csv"
    logger.info(f"Downloading ESPENI dataset from '{espeni_url}'.")

    hold_file = "data/espeni.csv"
    disable_progress = snakemake.config["run"].get("disable_progressbar", False)
    progress_retrieve(
        espeni_url,
        hold_file,
        disable=disable_progress
    )

    logger.info(f"ESPENI dataset downloaded to {hold_file}.")

    snapshots = pd.date_range(freq="h", **snakemake.params.snapshots)

    espeni = (
        pd.read_csv(hold_file, index_col=2, parse_dates=True)
        ["POWER_ESPENI_MW"]
        .resample("h").mean()
        .tz_localize(None)
        .loc[snapshots]
    )

    logger.warning(("Using ESPENI dataset for demand, which includes demand "
                 "met by embedded generation. This causes inaccuracy if "
                 "embedded generation capacities are not included in the model."))

    espeni.to_csv(snakemake.output[0])