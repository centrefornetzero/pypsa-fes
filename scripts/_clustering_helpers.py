# -*- coding: utf-8 -*-
"""
Functions for computing network clusters.
"""

__author__ = (
    "PyPSA Developers with some additions from Lukas Franken, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2023 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import logging

logger = logging.getLogger(__name__)

import networkx as nx
import numpy as np
import pandas as pd

from collections import namedtuple

from pypsa import io
from pypsa import Network
from pypsa.geo import haversine_pts
from pypsa.clustering.spatial import (
    Clustering,
    aggregateoneport,
    flatten_multiindex,
    normed_or_uniform,
    )


def _make_consense(component, attr):
    def consense(x):
        v = x.iat[0]
        assert (
            x == v
        ).all() or x.isnull().all(), "In {} cluster {} the values of attribute {} do not agree:\n{}".format(
            component, x.name, attr, x
        )
        return v

    return consense


def aggregategenerators(
    network,
    busmap,
    with_time=True,
    carriers=None,
    custom_strategies=dict(),
    aggregate_buses=None,
):

    if carriers is None:
        carriers = network.generators.carrier.unique()
    agg_carriers = network.generators.carrier.isin(carriers)

    """
    if aggregate_buses is None:
        aggregate_buses = busmap.astype(bool)
    aggregate_buses = aggregate_buses.loc[aggregate_buses].index
    agg_buses = network.generators.bus.isin(aggregate_buses)
    """
    if aggregate_buses is None:
        aggregate_buses = busmap.index.tolist()
    agg_buses = network.generators.bus.isin(aggregate_buses)

    gens_agg_b = agg_carriers + agg_buses

    attrs = network.components["Generator"]["attrs"]
    generators = network.generators.loc[gens_agg_b].assign(
        bus=lambda df: df.bus.map(busmap)
    )
    columns = (
        set(attrs.index[attrs.static & attrs.status.str.startswith("Input")])
        | {"weight"}
    ) & set(generators.columns) - {"control"}
    grouper = [generators.bus, generators.carrier]

    def normed_or_uniform(x):
        return (
            x / x.sum() if x.sum(skipna=False) > 0 else pd.Series(1.0 / len(x), x.index)
        )

    weighting = generators.weight.groupby(grouper, axis=0).transform(normed_or_uniform)
    generators["capital_cost"] *= weighting

    strategies = {
        "p_nom_max": pd.Series.min,
        "weight": pd.Series.sum,
        "p_nom": pd.Series.sum,
        "capital_cost": pd.Series.sum,
        "efficiency": pd.Series.mean,
        "ramp_limit_up": pd.Series.mean,
        "ramp_limit_down": pd.Series.mean,
        "ramp_limit_start_up": pd.Series.mean,
        "ramp_limit_shut_down": pd.Series.mean,
        "build_year": lambda x: 0,
        "lifetime": lambda x: np.inf,
    }
    strategies.update(custom_strategies)
    if strategies["p_nom_max"] is pd.Series.min:
        generators["p_nom_max"] /= weighting
    strategies.update(
        (attr, _make_consense("Generator", attr))
        for attr in columns.difference(strategies)
    )
    new_df = generators.groupby(grouper, axis=0).agg(strategies)
    new_df.index = flatten_multiindex(new_df.index).rename("name")

    new_df = pd.concat(
        [
            new_df,
            network.generators.loc[~gens_agg_b].assign(
                bus=lambda df: df.bus.map(busmap)
            ),
        ],
        axis=0,
        sort=False,
    )

    new_pnl = dict()
    if with_time:
        for attr, df in network.generators_t.items():
            pnl_gens_agg_b = df.columns.to_series().map(gens_agg_b)
            df_agg = df.loc[:, pnl_gens_agg_b]
            # If there are any generators to aggregate, do so and put
            # the time-varying data for all generators (aggregated and
            # non-aggregated) into `new_pnl`.
            if not df_agg.empty:
                if attr == "p_max_pu":
                    df_agg = df_agg.multiply(weighting.loc[df_agg.columns], axis=1)
                pnl_df = df_agg.groupby(grouper, axis=1).sum()
                pnl_df.columns = flatten_multiindex(pnl_df.columns).rename("name")
                new_pnl[attr] = pd.concat(
                    [df.loc[:, ~pnl_gens_agg_b], pnl_df], axis=1, sort=False
                )
            # Even if no generators are aggregated, we still need to
            # put the time-varying data for all non-aggregated
            # generators into `new_pnl`.
            elif not df.empty:
                new_pnl[attr] = df

    return new_df, new_pnl


def aggregatebuses(network, busmap, custom_strategies=dict()):
    attrs = network.components["Bus"]["attrs"]
    columns = set(
        attrs.index[attrs.static & attrs.status.str.startswith("Input")]
    ) & set(network.buses.columns)

    strategies = dict(
        x=pd.Series.mean,
        y=pd.Series.mean,
        v_nom=pd.Series.max,
        v_mag_pu_max=pd.Series.min,
        v_mag_pu_min=pd.Series.max,
    )
    strategies.update(
        (attr, _make_consense("Bus", attr)) for attr in columns.difference(strategies)
    )
    strategies.update(custom_strategies)

    return (
        network.buses.groupby(busmap)
        .agg(strategies)
        .reindex(
            columns=[
                f
                for f in network.buses.columns
                if f in columns or f in custom_strategies
            ]
        )
    )


def aggregatelines(network, buses, interlines, line_length_factor=1.0, with_time=True):
    # make sure all lines have same bus ordering
    positive_order = interlines.bus0_s < interlines.bus1_s
    interlines_p = interlines[positive_order]
    interlines_n = interlines[~positive_order].rename(
        columns={"bus0_s": "bus1_s", "bus1_s": "bus0_s"}
    )
    interlines_c = pd.concat((interlines_p, interlines_n), sort=False)

    attrs = network.components["Line"]["attrs"]
    columns = set(
        attrs.index[attrs.static & attrs.status.str.startswith("Input")]
    ).difference(("name", "bus0", "bus1"))

    consense = {
        attr: _make_consense("Bus", attr)
        for attr in (
            columns
            | {"sub_network"}
            - {
                "r",
                "x",
                "g",
                "b",
                "terrain_factor",
                "s_nom",
                "s_nom_min",
                "s_nom_max",
                "s_nom_extendable",
                "length",
                "v_ang_min",
                "v_ang_max",
            }
        )
    }

    def aggregatelinegroup(l):
        # l.name is a tuple of the groupby index (bus0_s, bus1_s)
        length_s = (
            haversine_pts(
                buses.loc[l.name[0], ["x", "y"]], buses.loc[l.name[1], ["x", "y"]]
            )
            * line_length_factor
        )
        v_nom_s = buses.loc[list(l.name), "v_nom"].max()

        voltage_factor = (np.asarray(network.buses.loc[l.bus0, "v_nom"]) / v_nom_s) ** 2
        non_zero_len = l.length != 0
        length_factor = (length_s / l.length[non_zero_len]).reindex(
            l.index, fill_value=1
        )

        data = dict(
            r=1.0 / (voltage_factor / (length_factor * l["r"])).sum(),
            x=1.0 / (voltage_factor / (length_factor * l["x"])).sum(),
            g=(voltage_factor * length_factor * l["g"]).sum(),
            b=(voltage_factor * length_factor * l["b"]).sum(),
            terrain_factor=l["terrain_factor"].mean(),
            s_max_pu=(l["s_max_pu"] * _normed(l["s_nom"])).sum(),
            s_nom=l["s_nom"].sum(),
            s_nom_min=l["s_nom_min"].sum(),
            s_nom_max=l["s_nom_max"].sum(),
            s_nom_extendable=l["s_nom_extendable"].any(),
            num_parallel=l["num_parallel"].sum(),
            capital_cost=(
                length_factor * _normed(l["s_nom"]) * l["capital_cost"]
            ).sum(),
            length=length_s,
            sub_network=consense["sub_network"](l["sub_network"]),
            v_ang_min=l["v_ang_min"].max(),
            v_ang_max=l["v_ang_max"].min(),
            lifetime=(l["lifetime"] * _normed(l["s_nom"])).sum(),
            build_year=(l["build_year"] * _normed(l["s_nom"])).sum(),
        )
        data.update((f, consense[f](l[f])) for f in columns.difference(data))
        return pd.Series(data, index=[f for f in l.columns if f in columns])

    lines = interlines_c.groupby(["bus0_s", "bus1_s"]).apply(aggregatelinegroup)
    lines["name"] = [str(i + 1) for i in range(len(lines))]

    linemap_p = interlines_p.join(lines["name"], on=["bus0_s", "bus1_s"])["name"]
    linemap_n = interlines_n.join(lines["name"], on=["bus0_s", "bus1_s"])["name"]
    linemap = pd.concat((linemap_p, linemap_n), sort=False)

    lines_t = dict()

    if with_time:
        for attr, df in network.lines_t.items():
            lines_agg_b = df.columns.to_series().map(linemap).dropna()
            df_agg = df.loc[:, lines_agg_b.index]
            if not df_agg.empty:
                if (attr == "s_max_pu") or (attr == "s_min_pu"):
                    weighting = network.lines.groupby(linemap).s_nom.apply(_normed)
                    df_agg = df_agg.multiply(weighting.loc[df_agg.columns], axis=1)
                pnl_df = df_agg.groupby(linemap, axis=1).sum()
                pnl_df.columns = flatten_multiindex(pnl_df.columns).rename("name")
                lines_t[attr] = pnl_df

    return lines, linemap_p, linemap_n, linemap, lines_t


def get_buses_linemap_and_lines(
    network, busmap, line_length_factor=1.0, bus_strategies=dict(), with_time=True
):
    # compute new buses
    buses = aggregatebuses(network, busmap, bus_strategies)

    lines = network.lines.assign(
        bus0_s=lambda df: df.bus0.map(busmap), bus1_s=lambda df: df.bus1.map(busmap)
    )

    # lines between different clusters
    interlines = lines.loc[lines["bus0_s"] != lines["bus1_s"]]
    lines, linemap_p, linemap_n, linemap, lines_t = aggregatelines(
        network, buses, interlines, line_length_factor, with_time
    )
    # network can be reduced to a set of isolated nodes in course of clustering (e.g. Rwanda)
    if lines.empty:
        lines_res = lines.drop(columns=["bus0_s", "bus1_s"]).reset_index(drop=True)
    else:
        lines_res = lines.reset_index().rename(
            columns={"bus0_s": "bus0", "bus1_s": "bus1"}, copy=False
        )
    return (
        buses,
        linemap,
        linemap_p,
        linemap_n,
        lines_res.set_index("name"),
        lines_t,
    )


Clustering = namedtuple(
    "Clustering",
    ["network", "busmap", "linemap", "linemap_positive", "linemap_negative"],
)


def get_clustering_from_busmap(
    network,
    busmap,
    with_time=True,
    line_length_factor=1.0,
    aggregate_generators_weighted=False,
    aggregate_one_ports={},
    aggregate_generators_carriers=None,
    scale_link_capital_costs=True,
    bus_strategies=dict(),
    one_port_strategies=dict(),
    generator_strategies=dict(),
    aggregate_generator_buses=None
):
    buses, linemap, linemap_p, linemap_n, lines, lines_t = get_buses_linemap_and_lines(
        network, busmap, line_length_factor, bus_strategies, with_time
    )

    network_c = Network()

    io.import_components_from_dataframe(network_c, buses, "Bus")
    io.import_components_from_dataframe(network_c, lines, "Line")

    # Carry forward global constraints to clustered network.
    network_c.global_constraints = network.global_constraints

    if with_time:
        network_c.set_snapshots(network.snapshots)
        network_c.snapshot_weightings = network.snapshot_weightings.copy()
        for attr, df in lines_t.items():
            if not df.empty:
                io.import_series_from_dataframe(network_c, df, "Line", attr)

    one_port_components = network.one_port_components.copy()

    if aggregate_generators_weighted:
        one_port_components.remove("Generator")
        generators, generators_pnl = aggregategenerators(
            network,
            busmap,
            with_time=with_time,
            carriers=aggregate_generators_carriers,
            custom_strategies=generator_strategies,
            aggregate_buses=aggregate_generator_buses,
        )
        io.import_components_from_dataframe(network_c, generators, "Generator")
        if with_time:
            for attr, df in generators_pnl.items():
                if not df.empty:
                    io.import_series_from_dataframe(network_c, df, "Generator", attr)

    for one_port in aggregate_one_ports:
        one_port_components.remove(one_port)
        new_df, new_pnl = aggregateoneport(
            network,
            busmap,
            component=one_port,
            with_time=with_time,
            custom_strategies=one_port_strategies.get(one_port, {}),
        )
        io.import_components_from_dataframe(network_c, new_df, one_port)
        for attr, df in new_pnl.items():
            io.import_series_from_dataframe(network_c, df, one_port, attr)

    ##
    # Collect remaining one ports

    for c in network.iterate_components(one_port_components):
        io.import_components_from_dataframe(
            network_c,
            c.df.assign(bus=c.df.bus.map(busmap)).dropna(subset=["bus"]),
            c.name,
        )

    if with_time:
        for c in network.iterate_components(one_port_components):
            for attr, df in c.pnl.items():
                if not df.empty:
                    io.import_series_from_dataframe(network_c, df, c.name, attr)

    new_links = (
        network.links.assign(
            bus0=network.links.bus0.map(busmap), bus1=network.links.bus1.map(busmap)
        )
        .dropna(subset=["bus0", "bus1"])
        .loc[lambda df: df.bus0 != df.bus1]
    )

    new_links["length"] = np.where(
        new_links.length.notnull() & (new_links.length > 0),
        line_length_factor
        * haversine_pts(
            buses.loc[new_links["bus0"], ["x", "y"]],
            buses.loc[new_links["bus1"], ["x", "y"]],
        ),
        0,
    )
    if scale_link_capital_costs:
        new_links["capital_cost"] *= (new_links.length / network.links.length).fillna(1)

    io.import_components_from_dataframe(network_c, new_links, "Link")

    if with_time:
        for attr, df in network.links_t.items():
            if not df.empty:
                io.import_series_from_dataframe(network_c, df, "Link", attr)

    io.import_components_from_dataframe(network_c, network.carriers, "Carrier")

    network_c.determine_network_topology()

    return Clustering(network_c, busmap, linemap, linemap_p, linemap_n)