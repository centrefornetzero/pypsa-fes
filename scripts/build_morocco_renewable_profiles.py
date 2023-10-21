# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Analogue to `scripts/build_renewable_profiles.py`, but for Morocco.

We disregards data beyond weather, and the regions's shapefile, hence
this script is much simpler than the original.

For details please refer to the original script.
"""

import logging

import json
import time
import atlite
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pypsa.geo import haversine
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from _helpers import configure_logging
from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_morocco_renewable_profiles", technology="onwind",
        )

    configure_logging(snakemake)

    nprocesses = int(snakemake.threads)
    noprogress = snakemake.config["run"].get("disable_progressbar", True)
    config = snakemake.config["renewable"][snakemake.wildcards.technology]

    resource = config["resource"]  # pv panel config / wind turbine config
    correction_factor = config.get("correction_factor", 1.0)
    capacity_per_sqkm = config["capacity_per_sqkm"]
    p_nom_max_meth = config.get("potential", "conservative")

    if correction_factor != 1.0:
        logger.info(f"correction_factor is set as {correction_factor}")

    cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)

    cutout = atlite.Cutout(snakemake.input.cutout)

    geoms = list()
    print(f'files for input: {snakemake.input[:4]}')

    for file in snakemake.input[:4]:
        print(file)

        with open(file) as f:
            data = json.load(f)

        for d in data['Data']:

            pts = gpd.points_from_xy(np.array(d['Data'])[0], np.array(d['Data'])[1])
            geoms.append(Polygon(pts))

    regions = (
        gpd.GeoDataFrame(
            geometry=[cascaded_union(geoms)],
            index=["MA0"]
        ).set_crs(epsg=4326).rename_axis("bus")
    )
    regions["country"] = ["MA"]
    regions[["x", "y"]] = regions.geometry.apply(lambda g: pd.Series(g.centroid.coords[0]))

    buses = regions.index

    epsg = 32629
    logger.warning(f"Using EPSG {epsg} processes.")
    res = config.get("excluder_resolution", 100)
    excluder = atlite.ExclusionContainer(crs=epsg, res=res)

    kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
    if noprogress:
        logger.info("Calculate landuse availabilities...")
        start = time.time()
        availability = cutout.availabilitymatrix(regions, excluder, **kwargs)
        duration = time.time() - start
        logger.info(f"Completed availability calculation ({duration:2.2f}s)")
    else:
        availability = cutout.availabilitymatrix(regions, excluder, **kwargs)

    area = cutout.grid.to_crs(epsg).area / 1e6
    area = xr.DataArray(
        area.values.reshape(cutout.shape), [cutout.coords["y"], cutout.coords["x"]]
    )

    potential = capacity_per_sqkm * availability.sum("bus") * area
    func = getattr(cutout, resource.pop("method"))
    resource["dask_kwargs"] = {"scheduler": client}
    capacity_factor = correction_factor * func(capacity_factor=True, **resource)
    layout = capacity_factor * area * capacity_per_sqkm
    profile, capacities = func(
        matrix=availability.stack(spatial=["y", "x"]),
        layout=layout,
        index=buses,
        per_unit=True,
        return_capacity=True,
        **resource,
    )

    logger.info(f"Calculating maximal capacity per bus (method '{p_nom_max_meth}')")
    if p_nom_max_meth == "simple":
        p_nom_max = capacity_per_sqkm * availability @ area
    elif p_nom_max_meth == "conservative":
        max_cap_factor = capacity_factor.where(availability != 0).max(["x", "y"])
        p_nom_max = capacities / max_cap_factor
    else:
        raise AssertionError(
            'Config key `potential` should be one of "simple" '
            f'(default) or "conservative", not "{p_nom_max_meth}"'
        )

    logger.info("Calculate average distances.")
    layoutmatrix = (layout * availability).stack(spatial=["y", "x"])

    coords = cutout.grid[["x", "y"]]
    bus_coords = regions[["x", "y"]]

    average_distance = []
    centre_of_mass = []

    for bus in buses:
        row = layoutmatrix.sel(bus=bus).data
        nz_b = row != 0
        row = row[nz_b]
        co = coords[nz_b]
        distances = haversine(bus_coords.loc[bus], co)
        average_distance.append((distances * (row / row.sum())).sum())
        centre_of_mass.append(co.values.T @ (row / row.sum()))

    average_distance = xr.DataArray(average_distance, [buses])
    centre_of_mass = xr.DataArray(centre_of_mass, [buses, ("spatial", ["x", "y"])])

    ds = xr.merge(
        [
            (correction_factor * profile).rename("profile"),
            capacities.rename("weight"),
            p_nom_max.rename("p_nom_max"),
            potential.rename("potential"),
            average_distance.rename("average_distance"),
        ]
    )

    # select only buses with some capacity and minimal capacity factor
    ds = ds.sel(
        bus=(
            (ds["profile"].mean("time") > config.get("min_p_max_pu", 0.0))
            & (ds["p_nom_max"] > config.get("min_p_nom_max", 0.0))
        )
    )

    if "clip_p_max_pu" in config:
        min_p_max_pu = config["clip_p_max_pu"]
        ds["profile"] = ds["profile"].where(ds["profile"] >= min_p_max_pu, 0)

    ds.to_netcdf(snakemake.output.profile)
    client.shutdown()