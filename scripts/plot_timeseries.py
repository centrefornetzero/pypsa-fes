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

    year = str(snakemake.wildcards["year"])
    
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
    axs[0].set_title(f"GB Emissions; {scenario_mapper[snakemake.wildcards.fes]}; {year}")

    plt.savefig(snakemake.output.emission_timeseries)
    plt.show()


def get_store_interaction(n, store_name):
    connect0 = n.links.query("bus0 == @store_name")
    connect1 = n.links.query("bus1 == @store_name")
    connect2 = n.links.query("bus2 == @store_name")

    connect_t = (
        pd.concat((
            n.links_t.p0[connect0.index],   
            n.links_t.p1[connect1.index],   
            n.links_t.p2[connect2.index]   
        ), axis=1)
        .mul(-1.)
        .groupby(n.links_t.p0.index.month).sum()
    )
    connect = pd.concat((connect0, connect1, connect2))

    connect_t = pd.concat((
        pd.DataFrame({
            "value": connect_t[connect.query("carrier == @carrier").index].sum(axis=1).mul(1e-6), # to MtCO2
            "carrier": carrier
            })
        for carrier in connect.carrier.unique()
    ), axis=0)

    connect_t["month"] = connect_t.index.map(lambda x: calendar.month_abbr[x])

    return connect_t


def make_co2_barplot(n):

    bar_kwargs = {
        "edgecolor": "black",
        "linewidth": 1,
        "alpha": 0.8,
        "x": "month",
        "y": "value",
        "hue": "carrier",
        "palette": "bright",
    }

    storing_t = get_store_interaction(n, "gb co2 stored")    
    emitting_t = get_store_interaction(n, "gb co2 atmosphere")

    _, axs = plt.subplots(1, 2, figsize=(16, 4))

    sns.barplot(
        data=emitting_t,
        ax=axs[0],
        **bar_kwargs
        )
        
    sns.barplot(
        data=storing_t,
        ax=axs[1],
        **bar_kwargs
        )

    for ax in axs:
        ax.set_xlabel("Month")
        ax.legend(loc=9)

    axs[0].set_ylabel("Monthly CO2 Emissions by Tech [MtCO2]")
    axs[1].set_ylabel("Monthly CO2 Removal by Tech [MtCO2]")

    total_stored = n.stores_t.e["gb co2 stored"].iloc[-1] * 1e-6
    total_emitted = n.stores_t.p["gb co2 atmosphere"].sum() * -1e-6

    for ax, value, meaning in zip(axs[::-1], [total_stored, total_emitted], ["Stored", "Emitted"]):
        ax.set_title(f"Total {meaning}: {value:.2f} MtCO2")

    plt.tight_layout()
    plt.savefig(snakemake.output.co2_barplot)
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
    tech_colors["grid battery"] = tech_colors["battery"]
    tech_colors["electricity demand"] = tech_colors["Electric load"]
    tech_colors["heat demand"] = tech_colors["CHP heat"]
    tech_colors["thermal inertia"] = tech_colors["nuclear"]
    tech_colors["intelligent EV charging"] = "#808080"
    tech_colors["intelligent EV discharging"] = "#CCCCCC"

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    # plot_emission_timeseries(n)    

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
                    logger.warning("Hardcoded data gathering of DAC.")
                    dac = c.df.loc[c.df.carrier == "DAC"]
                    subset = dac.loc[dac.bus2.isin(buses)]

                    outflow["DAC"] = - c.pnl.p2[subset.index].sum(axis=1)

                    dac = c.df.loc[c.df.carrier == "DAC"]

                    logger.warning("Hardcoded data gathering of thermal inertia.")
                    thermal_inertia_idx = c.df.loc[c.df.bus1.str.contains("thermal inertia")].index

                    thermal_flow = - c.pnl.p0[thermal_inertia_idx].sum(axis=1)

                    inflow["thermal inertia"] = np.maximum(thermal_flow.values, 0.)
                    outflow["thermal inertia"] = np.minimum(thermal_flow.values, 0.)


        if "bev" in snakemake.wildcards.flexopts:
            # split BEV charger into transfer into transport and into EV battery for
            ev_batteries = n.stores.loc[
                (n.stores.carrier == "battery storage") &
                (n.stores.location.isin(buses))
                ].index

            charging_eta = n.links.loc[n.links.carrier == "BEV charger"].efficiency.unique()
            if len(charging_eta) > 1:
                raise ValueError("Charging efficiency is not unique, code not built for this right now.")

            charging_eta = charging_eta[0]

            outflow["intelligent EV discharging"] = np.minimum(n.stores_t.p[ev_batteries].sum(axis=1), 0.) / charging_eta
            inflow["intelligent EV charging"] = np.maximum(n.stores_t.p[ev_batteries].sum(axis=1), 0.)
            inflow["intelligent EV charging"] -= inflow["V2G"]

            ev_transport = n.loads.loc[n.loads.carrier == "land transport EV"].index
            outflow["land transport EV"] = - n.loads_t.p_set[ev_transport].sum(axis=1)

            outflow.drop(columns=["BEV charger"], inplace=True)

        # add electricity demand
        outflow["electricity demand"] = - load

        heat_load_idx = n.loads.loc[
            (n.loads.carrier == "elec heat demand") &
            (n.buses.loc[n.loads.bus, "location"].isin(buses))
            ].index

        outflow["heat demand"] = - n.loads_t.p_set[heat_load_idx].sum(axis=1)
        if "heat pump" in outflow.columns:
            outflow.drop(columns="heat pump", inplace=True)

        both = inflow.loc[:, ((inflow > 0.).any() * (inflow < 0.).any())].columns
        outflow[both] = np.minimum(inflow[both].values, 0.)
        inflow[both] = np.maximum(inflow[both].values, 0.)

        if not both.empty:
            print(f"Added columns {', '.join(both.tolist())} to outflow.")

        both = outflow.loc[:, ((outflow > 0.).any() * (outflow < 0.).any())].columns
        inflow[both] = np.maximum(outflow[both].values, 0.)
        outflow[both] = np.minimum(outflow[both].values, 0.)

        if not both.empty:
            print(f"Added columns {', '.join(both.tolist())} to inflow.")

        #removing always zero inflows and outflows
        inflow = inflow.loc[:, (inflow != 0.).any()]
        outflow = outflow.loc[:, (outflow != 0.).any()]

        assert (inflow >= 0.).all().all(), "Inflow contains negative values."
        assert (outflow <= 0.).all().all(), "Outflow contains positive values."

        total_balance = abs((inflow.sum().sum() + outflow.sum().sum()) / inflow.sum().sum())
        if not np.allclose(total_balance, 0., atol=1e-3):
            logger.warning(f"Total imbalance in- and outflow {total_balance*100:.2f}% exceeds 0.1% for {target}.")

        inflow *= 1e-3
        outflow *= 1e-3

        total = inflow.sum(axis=1) + outflow.sum(axis=1)

        if target == "gb":
            inflow.to_csv(snakemake.output.timeseries_inflow)
            outflow.to_csv(snakemake.output.timeseries_outflow)

        # make whole year plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        sorting = inflow.std().sort_values().index.tolist()
        flexs = [carrier for carrier in inflow.columns if "flex" in carrier]
        for flex in flexs:
            sorting.remove(flex)
            sorting.append(flex)

        stackplot_to_ax(
            inflow.resample(freq).mean()[sorting],
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
            f"{ scenario_mapper[snakemake.wildcards.fes]};"
            f"{ snakemake.wildcards.year}"
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
            inflow.loc[inflow.index.month == month][sorting],
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
            f" {scenario_mapper[snakemake.wildcards.fes]};"
            f" {snakemake.wildcards.year};"
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

    make_co2_barplot(n)
