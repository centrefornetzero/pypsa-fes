# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Creates timeseries of generation profiles for desired regions

"""

import logging

logger = logging.getLogger(__name__)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
import pypsa
import seaborn as sns

from _helpers import override_component_attrs
from _fes_helpers import scenario_mapper
from make_summary import assign_carriers, assign_locations 
from plot_gb_validation import stackplot_to_ax


def plot_emission_timeseries(n):

    year = str(snakemake.wildcards["planning_horizons"])
    
    _, axs = plt.subplots(2, 1, figsize=(16, 10))

    regions = ["England", "Scotland", "GB"]
    bus_names = [
        snakemake.config["plotting"]["timeseries_groups"]["england"],
        snakemake.config["plotting"]["timeseries_groups"]["scotland"],
        "all",
    ]

    emittors = n.links.loc[n.links.bus2 == "gb co2 atmosphere"]

    line_kwargs = dict(
    )
    marker_kwargs = dict(
        marker="o",
        edgecolor="black",
        s=50,
        alpha=1.0,
        zorder=10,
    )

    linestyles = ["-", "--", "-."]

    for buses, region, ls in zip(bus_names, regions, linestyles):

        line_kwargs["linestyle"] = ls

        logger.info(f"Gathering emission timeseries for {region}...")

        if buses == "all":
            buses = pd.Index(n.buses.location.unique())
            buses = buses[buses.str.contains("GB")]

        local_emittors = emittors.loc[emittors.bus1.isin(buses)]

        emissions = (
            n.links_t
            .p2[local_emittors.index]
            .mul(-1)
            .mul(1e-6)
            .sum(axis=1)
        )

        emissions.cumsum().plot(ax=axs[0], label=region, **line_kwargs)
        ss = emissions.cumsum().iloc[::300]

        axs[0].scatter(ss.index, ss.values, **marker_kwargs)

        if region == "GB":

            by_month = emissions.groupby(emissions.index.month).sum()
            by_month.index = [calendar.month_name[i] for i in range(1, 13)]

            sns.barplot(
                x=by_month.index, 
                y=by_month.values,
                ax=axs[1],
                alpha=0.8,
                label="GB Monthly Emissions",
                edgecolor="black",
                linewidth=1,
                color="royalblue",
            )

    for ax in axs:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels = [label.replace("2022", year) for label in labels]
        labels = [label.replace("2021", str(int(year)-1)) for label in labels]
        ax.set_xticklabels(labels)

    axs[0].legend()
    axs[0].set_ylabel("Cumulative Emissions [MtCO2]")
    axs[1].set_ylabel("Monthly Emissions GB [MtCO2]")
    axs[0].set_xlabel("Time")
    axs[1].set_xlabel("Month")
    axs[0].set_title(f"GB Emissions; {scenario_mapper[snakemake.wildcards.fes_scenario]}; {year}")

    plt.savefig(snakemake.output.emission_timeseries)
    plt.show()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("plot_timeseries")

    tech_colors = snakemake.config["plotting"]["tech_colors"]
    tech_colors["wind"] = tech_colors["onwind"]
    tech_colors["H2 dispatch"] = tech_colors["H2 Fuel Cell"]
    tech_colors["H2 charge"] = tech_colors["H2 Electrolysis"]
    tech_colors["battery dispatch"] = tech_colors["battery"]
    tech_colors["battery charge"] = tech_colors["BEV charger"]
    tech_colors["DC"] = tech_colors["HVDC links"]
    tech_colors["AC"] = tech_colors["AC-AC"]
    tech_colors["GAS CCS"] = tech_colors["power-to-H2"]

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    plot_emission_timeseries(n)    

    assign_carriers(n)
    assign_locations(n)

    freq = snakemake.config["plotting"]["timeseries_freq"]
    month = snakemake.config["plotting"]["timeseries_month"]
    
    target_files = ["gb", "scotland", "england"]
    bus_names = ["all",
            snakemake.config["plotting"]["timeseries_groups"]["scotland"],
            snakemake.config["plotting"]["timeseries_groups"]["england"]
    ]

    for buses, target in zip(bus_names, target_files):

        logger.info(f"Plotting timeseries for {target}...")

        if buses == "all":
            buses = pd.Index(n.buses.location.unique())
            buses = buses[buses.str.contains("GB")]

        def intersection(lst1, lst2):
            return list(set(lst1) & set(lst2))

        load = n.loads_t.p_set.loc[:, intersection(n.loads.index, buses)].sum(axis=1)
        inflow = pd.DataFrame(index=n.snapshots)
        outflow = pd.DataFrame(index=n.snapshots)

        for c in n.iterate_components(n.one_port_components | n.branch_components):

            if c.name == "Load":
                continue

            if c.name in ["Generator", "StorageUnit"]: 
                idx = c.df.loc[c.df.bus.isin(buses)].index

                c_energies = (
                    c.pnl.p
                    .loc[:, idx]
                    .multiply(n.snapshot_weightings.generators, axis=0)
                )

                for carrier in c.df.loc[idx, "carrier"].unique():
                    cols = c.df.loc[idx].loc[c.df.loc[idx, "carrier"] == carrier].index

                    inflow[carrier] = c_energies.loc[:, cols].sum(axis=1)

            elif c.name in ["Link", "Line"]:

                inout = ((c.df.bus0.isin(buses)) ^ (c.df.bus1.isin(buses))).rename("choice")

                subset = c.df.loc[inout]
                asbus0 = subset.loc[subset.bus0.isin(buses)].index
                asbus1 = subset.loc[subset.bus1.isin(buses)].index

                inout = pd.concat((
                    - c.pnl.p0[asbus0],
                    - c.pnl.p1[asbus1]
                ), axis=1)

                for carrier in c.df.loc[subset.index, "carrier"].unique():
                    flow = inout[subset.loc[subset.carrier == carrier].index].sum(axis=1)

                    inflow[carrier] = np.maximum(flow.values, 0.)
                    outflow[carrier] = np.minimum(flow.values, 0.)

                if c.name == "Link":
                    logger.warning("Hardcoded data gathering of DAC")
                    dac = c.df.loc[c.df.carrier == "DAC"]
                    subset = dac.loc[dac.bus2.isin(buses)]

                    outflow["DAC"] = - c.pnl.p2[subset.index].sum(axis=1)


        load *= 1e-3
        inflow *= 1e-3
        outflow *= 1e-3

        total = inflow.sum(axis=1) + outflow.sum(axis=1) - load

        # make whole year plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        stackplot_to_ax(
            # inflow.resample(freq).mean(),
            inflow.resample(freq).mean()[inflow.std().sort_values().index],
            ax=ax,
            color_mapper=tech_colors,
            )

        if not outflow.empty:
            stackplot_to_ax(
                outflow.resample(freq).mean(),
                ax=ax,
                color_mapper=tech_colors,
                )

        ax.plot(total.index,
                total,
                color="black",
                label="Kirchhoff Check",
                linestyle="--",
                )        

        ax.set_ylabel("Generation (GW)")

        title = (
            f"{target.upper()};"
            f"{ scenario_mapper[snakemake.wildcards.fes_scenario]};"
            f"{ snakemake.wildcards.planning_horizons}"
        )        
        ax.set_title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0.75, -0.05),
            fancybox=True,
            shadow=True,
            ncol=5,
            )

        plt.tight_layout()
        plt.savefig(snakemake.output[f"timeseries_{target}_year"])
        plt.show()

        # make plot of one month
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        stackplot_to_ax(
            inflow.loc[inflow.index.month == month][inflow.std().sort_values().index],
            ax=ax,
            color_mapper=tech_colors,
            )

        if not outflow.empty:
            stackplot_to_ax(
                outflow.loc[inflow.index.month == month],
                ax=ax,
                color_mapper=tech_colors,
                )

        ax.plot(total.loc[inflow.index.month == month].index,
                total.loc[inflow.index.month == month],
                color="black",
                label="Kirchhoff Check",
                linestyle="--",
                linewidth=1.,
                )

        ax.set_ylabel("Generation (GW)")
        title = (
            f"{target.upper()};"
            f" {scenario_mapper[snakemake.wildcards.fes_scenario]};"
            f" {snakemake.wildcards.planning_horizons};"
            f" {calendar.month_name[month]}"
        )

        ax.set_title(title)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0.75, -0.05),
            fancybox=True,
            shadow=True,
            ncol=5,
            )

        plt.tight_layout() 

        plt.savefig(snakemake.output[f"timeseries_{target}_short"])
        plt.show()
