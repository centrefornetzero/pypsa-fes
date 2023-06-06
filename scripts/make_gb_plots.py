# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023 Lukas Franken
#
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import pypsa
import pyomo.environ as po
import seaborn as sns
import geopandas as gpd
from cartopy.crs import ccrs
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from pypsa.plot import add_legend_patches

import logging

logger = logging.getLogger(__name__)


def build_nice_names(cfg):
    nice_names = cfg["plotting"]["nice_names"]

    nice_names["biomass"] = "Biomass"
    nice_names["coal"] = "Hard Coal"
    nice_names["lignite"] = "Lignite"
    nice_names["nuclear"] = "Nuclear"
    nice_names["oil"] = "Oil"

    return nice_names


def build_map_opts(cfg):
    map_opts = cfg["plotting"]["map"]
    map_opts["color_geomap"]["ocean"] = "powderblue"
    map_opts["color_geomap"]["land"] = "oldlace"

    return map_opts


def plot_regions(regions_file, ax=None, savefile=None):
    """Shows network regions in delightful coloruring"""

    regions = gpd.read_file(regions_file)
    regions["values"] = np.random.uniform(size=len(regions))

    show_plot = False
    if ax is None:
        show_plot = True
        _, ax = plt.subplots(1, 1, figsize=(4, 6))

    regions.plot(ax=ax, edgecolor="k", linewidth=0.75, column="values", cmap="prism")

    if show_plot:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if savefile is not None:
            plt.savefig(savefile)
        plt.show()


def plot_gb_network(n, cfg, ax=None, savefile=None):

    map_opts = build_map_opts(cfg)
    nice_names = build_nice_names(cfg)
    tech_colors = cfg["plotting"]["tech_colors"]

    total_generation = pd.DataFrame(index=n.buses.loc[n.buses.carrier == "AC"].index)

    for carrier in n.generators.carrier.value_counts().index:
        gens = n.generators.loc[n.generators.carrier == carrier].index
        
        totals = n.generators_t.p[gens].sum()
        
        totals.index = [name[:-(len(carrier)+1)] for name in totals.index]
             
        total_generation[carrier] = np.zeros(len(total_generation))
        total_generation.loc[totals.index, carrier] = totals

    total_generation = total_generation.stack()
    # bus_sizes = total_generation.groupby(level=0).sum()

    show_plot = False
    if ax is None:
        show_plot = True
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection":ccrs.PlateCarree()})

    n.plot(
        ax=ax,
        bus_sizes=total_generation / total_generation.groupby(level=0).sum().max() * 2.,
        bus_colors=tech_colors,
        **map_opts
    )

    carriers = total_generation.groupby(level=1).sum().index
    colors = [tech_colors[c] for c in carriers]
    labels = [nice_names[carrier] for carrier in carriers]

    add_legend_patches(
        ax,
        colors,
        labels,
        legend_kw={},
    )
    
    ax.set_xlim(-8, 2)
    ax.set_ylim(48, 60)

    if savefile:
        plt.savefig(savefile)

    if show_plot: plt.show()


def plot_gb_generation_timeseries(n, cfg, ax=None, savefile=None):

    nice_names = build_nice_names(cfg)
    tech_colors = cfg["plotting"]["tech_colors"]

    buses = n.buses.loc[n.buses.country == "GB"]
    buses = buses.loc[buses.carrier == "AC"]

    lower_index = 5000
    upper_index = 5300

    total_generation = pd.DataFrame(index=n.generators_t.p.index)

    for carrier in n.generators.carrier.unique():
        subset = n.generators.loc[n.generators.carrier == carrier].index
        subset = [col for col in subset if "GB" in col]
        total_generation[carrier] = n.generators_t.p[subset].sum(axis=1)

    show_results = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        show_results = True

    total_generation_subset = total_generation.iloc[lower_index:upper_index]
    total_generation_subset = total_generation_subset[total_generation_subset.std().sort_values(ascending=True).index].multiply(1e-3)

    ax.stackplot(total_generation_subset.index,
                list(total_generation_subset.values.T), 
                edgecolor="grey",
                linewidth=.8,
                alpha=0.85,
                colors=pd.Series(tech_colors).loc[total_generation_subset.columns].values,
                labels=[nice_names[carrier] for carrier in total_generation_subset.columns])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
    ax.set_ylabel("Generation [GW]")
    ax.set_title("Accumulated Generation in GB")
    ax.set_xlim(total_generation_subset.index[0], total_generation_subset.index[-1])

    if savefile:
        plt.savefig(savefile)
    if show_results:
        plt.show()


def make_total_generation_barplot(n, cfg, ax=None, savefile=None):

    nice_names = build_nice_names(cfg)
    tech_colors = cfg["plotting"]["tech_colors"]

    total_generation = pd.DataFrame(index=n.generators_t.p.index)

    for carrier in n.generators.carrier.unique():
        subset = n.generators.loc[n.generators.carrier == carrier].index
        subset = [col for col in subset if "GB" in col]
        total_generation[carrier] = n.generators_t.p[subset].sum(axis=1)

    show_results = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        show_results = True

    total_total = (total_generation.sum() * 1e-6).sort_values(ascending=False)

    total_total.plot.bar(ax=ax, edgecolor="k", linewidth=0.75, color="darkred", alpha=0.8)
    ax.set_xticklabels([nice_names[name] for name in total_total.index])

    if savefile:
        plt.savefig(savefile)
    if show_results:
        ax.set_ylabel("Total Generation [TWh]")
        plt.show()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("make_summary")

    logging.basicConfig(level=snakemake.config["logging"]["level"])

    plot_regions(snakemake.input["regions_onshore"],
                 savefile=snakemake.output["regions_map"])
    plot_gb_network(snakemake.input["network"],
                    savefile=snakemake.output["capacity_expansion"])
    plot_gb_generation_timeseries(snakemake.input["network"],
                    savefile=snakemake.output["generation_timeseries"])
    make_total_generation_barplot(snakemake.input["network"],
                    savefile=snakemake.output["total_generation"])