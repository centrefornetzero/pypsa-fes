# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Create cutouts with `atlite <https://atlite.readthedocs.io/en/latest/>`_.

Adapted from `scripts/build_cutout.py`, please refer to the original script for
more information.

Relevant Settings
-----------------

.. code:: yaml

    atlite:
        nprocesses:
        cutouts:
            {cutout}:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`atlite_cf`

"""

import logging

import json
import atlite
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from shapely.ops import cascaded_union
from shapely.geometry import Polygon
from _helpers import configure_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_morroco_cutout", cutout="morocco-2019-era5")

    configure_logging(snakemake)

    cutout_params = snakemake.config["morocco_atlite"][snakemake.wildcards.morocco_cutout]

    snapshots = pd.date_range(freq="h", **snakemake.config["snapshots"])
    time = [snapshots[0], snapshots[-1]]
    cutout_params["time"] = slice(*cutout_params.get("time", time))

    if {"x", "y", "bounds"}.isdisjoint(cutout_params):
        print("We are in the first case")

        geoms = list()
        for file in snakemake.input:

            data = json.load(Path(file).open())

            for d in data['Data']:

                pts = gpd.points_from_xy(np.array(d['Data'])[0], np.array(d['Data'])[1])
                geoms.append(Polygon(pts))

        regions = gpd.GeoDataFrame(geometry=[cascaded_union(geoms)]).set_crs(epsg=4326)

        print("morocco shapes", regions)
        regions.plot()

        # Determine the bounds from bus regions with a buffer of two grid cells
        d = max(cutout_params.get("dx", 0.25), cutout_params.get("dy", 0.25)) * 2
        cutout_params["bounds"] = regions.total_bounds + [-d, -d, d, d]

    elif {"x", "y"}.issubset(cutout_params):
        print("We are in the second case")
        cutout_params["x"] = slice(*cutout_params["x"])
        cutout_params["y"] = slice(*cutout_params["y"])

    logging.info(f"Preparing Morocco cutout with parameters {cutout_params}.")
    features = cutout_params.pop("features", None)
    cutout = atlite.Cutout(snakemake.output[0], **cutout_params)
    cutout.prepare(features=features)
