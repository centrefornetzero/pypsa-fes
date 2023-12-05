# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:21:03 2023

@author: Claire Halloran, University of Oxford

This script builds hourly air and soil temperature profiles 
for each clustered network region. Inspired by build_temperature_profiles.py
in PyPSA-Eur v. 0.8.0.
"""

import atlite
import logging
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rio
import progressbar as pgb
from dask.distributed import Client, LocalCluster
from _helpers import configure_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_temperature_profiles",
            simpl='',
            clusters=39
            )
    configure_logging(snakemake)
    pgb.streams.wrap_stderr()

    nprocesses = int(snakemake.threads)
    noprogress = not snakemake.params["atlite"].get("show_progress", False)

    cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)
    
    time = pd.date_range(freq="h", **snakemake.params["snapshots"])
    cutout = atlite.Cutout(snakemake.input.cutout).sel(time=time)

    regions = gpd.read_file(snakemake.input.regions)

    assert not regions.empty, (
        f"List of regions in {snakemake.input.regions} is empty, please "
        "disable the corresponding renewable technology"
    )
    # do not pull up, set_index does not work if geo dataframe is empty
    regions = regions.set_index("name").rename_axis("bus")
    buses = regions.index
    # import population raster
    population = rio.open_rasterio(snakemake.input.population)
    population.rio.set_spatial_dims(x_dim='x',y_dim='y')

    cutout_rio = cutout.data
    cutout_rio = cutout_rio.rio.write_crs('EPSG:4326')
    # transform to same CRS and resolution as cutout
    population_match = population.rio.reproject_match(cutout_rio,
                                                      resampling = rasterio.enums.Resampling.sum)
    # # change large negative values to nan
    population_match = population_match.squeeze().drop('band')
    population_match = population_match.where(population_match>0.)

    if snakemake.params['heating']['single_GB_temperature']:
        # calculate population-weighted national average hourly air and soil temperature
        total_population = population_match.sum(dim=['x', 'y'])
        weighted_temperature = (cutout.data['temperature'] * population_match).sum(dim=['x', 'y']) / total_population
        weighted_soil_temperature = (cutout.data['soil temperature'] * population_match).sum(dim=['x', 'y']) / total_population

        # use mask of population to replace temperature within Britain with average
        cutout.data['temperature'] = cutout.data['temperature'].where(population_match.isnull(),weighted_temperature)
        cutout.data['soil temperature'] = cutout.data['soil temperature'].where(population_match.isnull(),weighted_soil_temperature)

   # from 0.8.0 build_temperature_profiles.py
   # do not pull up-- can't fill with 0s until after replacing temperatures
    population_match = population_match.fillna(0.)

    I = cutout.indicatormatrix(regions)

    stacked_pop = population_match.stack(spatial=("y", "x"))

    M = I.T.dot(np.diag(I.dot(stacked_pop)))

    nonzero_sum = M.sum(axis=0, keepdims=True)
    nonzero_sum[nonzero_sum == 0.0] = 1.0
    M_tilde = M / nonzero_sum
    M_tilde = np.nan_to_num(M_tilde,0.)
    # population-weighted average temperature
    temp_air = cutout.temperature(matrix=M_tilde.T,
                                  index=regions.index,
                                  )
    temp_air.to_netcdf(snakemake.output.temp_air)

    temp_ground = cutout.soil_temperature(matrix=M_tilde.T,
                                  index=regions.index,
                                  )

    temp_ground.to_netcdf(snakemake.output.temp_ground)