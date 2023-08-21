# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Creates plots for optimised network topologies, including electricity, gas and
hydrogen networks, and regional generation, storage and conversion capacities
built.

This rule plots a map of the network with technology capacities at the
nodes.
"""

import logging

logger = logging.getLogger(__name__)

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from _helpers import override_component_attrs

from cartopy.io import shapereader
from make_summary import assign_carriers
from plot_summary import preferred_order, rename_techs
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from shapely.ops import unary_union

plt.style.use(["ggplot", "matplotlibrc"])


def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis", "methanation", "helmeth", "H2 liquefaction"]:
        return "power-to-gas"
    elif tech == "H2":
        return "H2 storage"
    elif tech in ["NH3", "Haber-Bosch", "ammonia cracker", "ammonia store"]:
        return "ammonia"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    # elif "solar" in tech:
    #     return "solar"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "CC" in tech or "sequestration" in tech:
        return "CCS"
    else:
        return tech


def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1:
                continue
            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = names.str[:i]


def plot_map(
    network,
    regions,
    components=["links", "stores", "storage_units", "generators"],
    bus_size_factor=1.7e10,
    transmission=True,
    with_legend=True,
):
    tech_colors = snakemake.config["plotting"]["tech_colors"]
    tech_colors["thermal generation CC"] = tech_colors["H2 Fuel Cell"]
    tech_colors["grid battery"] = tech_colors["battery"]
    tech_colors["thermal generation unabated"] = tech_colors["CCGT"]

    n = network.copy()
    assign_location(n)
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    costs = pd.DataFrame(index=n.buses.index)

    for comp in components:
        df_c = getattr(n, comp)

        if df_c.empty:
            continue

        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)

        attr_opt = "e_nom_opt" if comp == "stores" else "p_nom_opt"
        attr_fixed = "e_nom" if comp == "stores" else "p_nom"
        col = "e_nom_extendable" if comp == "stores" else "p_nom_extendable"

        capacity = pd.concat((
            df_c.loc[df_c[col], attr_opt],
            df_c.loc[~df_c[col], attr_fixed],
        ))
        
        costs_c = (
            capacity
            .groupby([df_c.location, df_c.nice_group])
            .sum()
            .unstack()
            .fillna(0.0)
        )
        costs = pd.concat([costs, costs_c], axis=1)

        logger.debug(f"{comp}, {costs}")

    costs = costs.groupby(costs.columns, axis=1).sum()

    costs *= 1e5

    costs.drop(list(costs.columns[(costs == 0.0).all()]), axis=1, inplace=True)

    new_columns = preferred_order.intersection(costs.columns).append(
        costs.columns.difference(preferred_order)
    )
    costs = costs[new_columns]

    for item in new_columns:
        if item not in tech_colors:
            logger.warning(f"{item} not in config/plotting/tech_colors")

    costs = costs.stack()  # .sort_index()

    # hack because impossible to drop buses...
    eu_location = snakemake.config["plotting"].get(
        "eu_node_location", dict(x=-5.5, y=46)
    )
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]

    n.links.drop(
        n.links.index[(n.links.carrier != "DC") & (n.links.carrier != "B2B")],
        inplace=True,
    )

    # drop non-bus
    to_drop = costs.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        costs.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    costs.index = pd.MultiIndex.from_tuples(costs.index.values)

    threshold = 100e6  # 100 mEUR/a
    carriers = costs.groupby(level=1).sum()
    carriers = carriers.where(carriers > threshold).dropna()
    carriers = list(carriers.index)

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.0
    line_upper_threshold = 1e4
    # line_lower_threshold = 0.
    # line_upper_threshold = np.inf

    linewidth_factor = 2.5e2
    ac_color = "rosybrown"
    dc_color = "darkseagreen"

    if snakemake.wildcards["ll"] == "v1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        title = "added grid"

        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "current grid"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        title = "added grid"

        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "total grid"

    # line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
    # link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)
    line_widths = line_widths.clip(line_lower_threshold, line_lower_threshold+1.)
    link_widths = link_widths.clip(line_lower_threshold, line_lower_threshold+1.)

    # line_widths = line_widths.replace(line_lower_threshold, 0)
    # link_widths = link_widths.replace(line_lower_threshold, 0)

    proj = ccrs.EqualEarth()

    fig, ax = plt.subplots(subplot_kw={"projection": proj})
    fig.set_size_inches(12, 10)

    regions = regions.to_crs(proj.proj4_init)

    carrier_grouping = snakemake.config["flexibility"]["map_grouping"]

    df = costs.unstack()

    interesting = np.array(list(carrier_grouping.values()), dtype=list).sum()
    df = df[df.columns.intersection(interesting)]

    for target, source in carrier_grouping.items():
        inter = df.columns.intersection(source).tolist()
        df[target] = df[inter].sum(axis=1)

        if target in inter:
            inter.remove(target)

        df.drop(columns=inter, inplace=True)
    
    costs = df.stack()

    for col in costs.unstack().columns:
        if col not in tech_colors:
            logger.warning(f"{col} not in config/plotting/tech_colors")
            tech_colors[col] = "grey"

    map_opts["color_geomap"] = False

    if not with_legend:
        map_opts["boundaries"][1] -= 1.
        
    n.plot(
        bus_sizes=costs / bus_size_factor,
        bus_colors=tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,
        **map_opts,
    )

    regions.plot(
        ax=ax,
        color="palegoldenrod",
        linewidths=1,
        edgecolor="k",
        legend=True,
        legend_kwds={
            "label": "Value",
            "shrink": 0.7,
            "extend": "max",
        },
    )

    # get country borders
    resolution = '10m'
    category = 'cultural'
    name = 'admin_0_countries'

    shpfilename = shapereader.natural_earth(resolution, category, name)

    # read the shapefile using geopandas
    df = gpd.read_file(shpfilename)

    other_countries = df.loc[df['ADMIN'].isin([
        "Ireland", "France", "Belgium", "Netherlands", "United Kingdom"
        ])]

    other_shapes = other_countries['geometry']
    other_shapes.index = other_countries["ADMIN"]
    other_shapes = other_shapes.to_crs(proj.proj4_init)

    uk_shape = gpd.GeoDataFrame(geometry=list(other_shapes.loc["United Kingdom"]))
    uk_shape = uk_shape.set_crs(proj.proj4_init)
    ni_shape = uk_shape.drop(uk_shape.loc[uk_shape.sjoin(regions).index.unique()].index)
    
    other_shapes.at["United Kingdom"] = unary_union(
        ni_shape.geometry.tolist())

    other_shapes.plot(
        ax=ax,
        color="white",
        linewidths=1,
        edgecolor="k",
    )

    """
    sizes = [20, 10, 5]
    labels = [f"{s} bEUR/a" for s in sizes]
    sizes = [s / bus_size_factor * 1e9 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.01, 1.06),
        labelspacing=0.8,
        frameon=False,
        handletextpad=0,
        title="system cost",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [10, 5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.27, 1.06),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
        title=title,
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="lightgrey"), legend_kw=legend_kw
    )
    """

    legend_kw = dict(
        # bbox_to_anchor=(1.52, 1.04),
        # frameon=False,
        frameon=True,
    )

    if with_legend:
        carriers = costs.index.get_level_values(1).unique().tolist()
        colors = [tech_colors[c] for c in carriers] + [ac_color, dc_color]
        labels = carriers + ["HVAC line", "HVDC link"]

        add_legend_patches(
            ax,
            colors,
            labels,
            legend_kw=legend_kw,
        )

    fig.savefig(snakemake.output.map, transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_network",
            simpl="",
            opts="",
            clusters="5",
            ll="v1.5",
            sector_opts="CO2L0-1H-T-H-B-I-A-solar+p3-dist1",
            planning_horizons="2030",
        )

    logging.basicConfig(level=snakemake.config["logging"]["level"])

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    regions = gpd.read_file(snakemake.input.regions).set_index("Name_1")
    regions = regions.loc[regions.index.str.contains("Z")]

    map_opts = snakemake.config["plotting"]["map"]

    if map_opts["boundaries"] is None:
        map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]

    plot_map(
        n,
        regions,
        components=["generators", "links", "stores", "storage_units"],
        bus_size_factor=2e10,
        transmission=False,
        with_legend=True,
    )