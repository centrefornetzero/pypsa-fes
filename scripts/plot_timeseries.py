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
import calendar
import seaborn as sns

from collections.abc import Iterable

from _fes_helpers import scenario_mapper
from make_summary import assign_carriers, assign_locations 
from plot_gb_validation import stackplot_to_ax


carrier_grouper = {
    "wind": ["offwind-ac", "offwind-dc", "onwind"],
    "solar": ["solar"],
    "hydro generation": ["hydro", "ror"],
    "hydro storage": ["PHS"],
    "thermal generation unabated": ["OCGT", "CCGT", "lignite", "coal"],
    "thermal generation CC": ["allam", "biomass"],
    "event flexibility": ["regular flex", "winter flex"], 
    "smart heat pump": ["thermal inertia"],
    "smart EV charger": ["intelligent EV charging", "intelligent EV discharging"],
    "vehicle to grid": ["V2G"],
    "interconnector": ["DC"],
    "direct air capture": ["DAC"],
}

demand_renaming = {
    "transport demand": ["land transport EV", "BEV charger"],
}

flex_grouping = {
    "load shifting": ["smart heat pump", "smart EV charger"],
    "demand reduction": ["turndown events", "vehicle to grid"],
}

demands = ["other electricity demand", "heat demand", "transport demand"][::-1]
generators = [
    "wind",
    "solar",
    "hydropower",
    "thermal generation unabated",
    "thermal generation CC",
    ]
shifters = [
    "hydro storage",
    "grid battery",
    "smart heat pump",
    "smart EV charger",
]

dropcols = ["direct air capture", "DAC"]


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


def get_timeseries_subset(*args, timeseries_mode="month", config=None):

    assert args, "No dataframes passed"

    assert sum(
        [isinstance(arg, pd.DataFrame) or
         isinstance(arg, pd.Series) for arg in args]) == len(args), "Only DataFrames can be passed as argument"

    if (mode := timeseries_mode) in ["month", "year"]:
        freq = config["flexibility"]["timeseries_params"][mode].get("freq", "1H")
        month = config["flexibility"]["timeseries_params"][mode].get("month", range(1, 13))

        if not isinstance(month, Iterable):
            month = [month]
        
        s = args[0].index[args[0].index.month.isin(month)]

        if mode == "month":
            logger.warning("Currently only shows two weeks in mode 'month'.")
            s = s[s < s[0] + pd.Timedelta(days=14)]        

        args = list(map(lambda x: x.loc[s].resample(freq).mean(), args))

    
    elif mode in ["shortweek", "longweek"]:

        frame = config["flexibility"]["timeseries_params"][mode]

        start = pd.Timestamp(
            year=2022,
            month=frame["start_month"],
            day=frame["start_day"]
        )

        end = pd.Timestamp(
            year=2022,
            month=frame["end_month"],
            day=frame["end_day"]
        )

        s = args[0].index[(args[0].index >= start) & (args[0].index <= end)]

        args = list(map(lambda x: x.loc[s], args))

    else:
        raise ValueError(f"Unknown mode {mode}, should be one of 'month', 'year', 'shortweek', 'longweek'")
    
    return args


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("plot_timeseries")

    config = snakemake.config
    tech_colors = config["plotting"]["tech_colors"]
    tech_colors["wind"] = tech_colors["onwind"]
    tech_colors["H2 dispatch"] = tech_colors["H2 Fuel Cell"]
    tech_colors["H2 charge"] = tech_colors["H2 Electrolysis"]
    tech_colors["battery dispatch"] = tech_colors["battery"]
    tech_colors["battery charge"] = tech_colors["BEV charger"]
    tech_colors["DC"] = tech_colors["HVDC links"]
    tech_colors["AC"] = tech_colors["AC-AC"]
    tech_colors["GAS CCS"] = tech_colors["power-to-H2"]
    tech_colors["grid battery"] = tech_colors["battery"]
    tech_colors["other electricity demand"] = tech_colors["Electric load"]
    tech_colors["heat demand"] = tech_colors["CHP heat"]
    tech_colors["transport demand"] = tech_colors["BEV charger"]
    tech_colors["thermal inertia"] = tech_colors["nuclear"]
    tech_colors["intelligent EV charging"] = "#808080"
    tech_colors["intelligent EV discharging"] = "#CCCCCC"
    tech_colors["turndown events"] = tech_colors["winter flex"]
    tech_colors["hydro generation"] = tech_colors["ror"]
    tech_colors["hydro storage"] = tech_colors["PHS"]
    tech_colors["ev battery"] = tech_colors["home battery"]
    tech_colors["residential building heat inertia"] = tech_colors["nuclear"]
    tech_colors["battery discharger"] = tech_colors["H2 dispatch"]

    
    for key, value in carrier_grouper.items():
        tech_colors[key] = tech_colors[value[0]]
    for key, value in flex_grouping.items():
        tech_colors[key] = tech_colors[value[0]]

    inflow = pd.read_csv(snakemake.input.inflow, index_col=0, parse_dates=True)
    outflow = pd.read_csv(snakemake.input.outflow, index_col=0, parse_dates=True)
    
    # def get_timeseries_subset(*args , timeseries_mode="month", config=None):

    ts_mode = snakemake.wildcards["timeseries_mode"]

    logger.info(f"Found time series mode {ts_mode}.")

    inflow, outflow = get_timeseries_subset(inflow, outflow,
        timeseries_mode=ts_mode,
        config=snakemake.config,
        )

    def group_by_dict(df, grouper):
        for key, value in grouper.items():

            if len(value) == 1 and key == value[0]:
                continue

            if len(overlap := df.columns.intersection(value)) > 0:
                df[key] = df[overlap].sum(axis=1)
                df.drop(overlap, axis=1, inplace=True)

        return df

    print("inflow cols")
    print(inflow.columns)
    print("outflow cols")
    print(outflow.columns)
    print("==========================")

    if config["flexibility"]["timeseries_params"]["do_group"]:
        
        inflow = group_by_dict(inflow, carrier_grouper)
        outflow = group_by_dict(outflow, carrier_grouper)

        if config["flexibility"]["timeseries_params"]["do_group_flex"]:

            inflow = group_by_dict(inflow, flex_grouping)
            outflow = group_by_dict(outflow, flex_grouping)

    else:
        assert not config["flexibility"]["timeseries_params"]["do_group_flex"], (
            "Cannot group flexibility without grouping technologies")
    
    outflow = group_by_dict(outflow, demand_renaming)

    print("==========================")
    print("inflow cols")
    print(inflow.columns)
    print("outflow cols")
    print(outflow.columns)

    inflow.drop(inflow.columns.intersection(dropcols), axis=1, inplace=True)
    outflow.drop(outflow.columns.intersection(dropcols), axis=1, inplace=True)

    total = inflow.sum(axis=1) + outflow.sum(axis=1)

    # make whole year plot
    fig, axs = plt.subplots(4, 1, figsize=(16, 16))

    inflow_sorting = inflow.mean().sort_values(ascending=False).index.tolist()

    stackplot_to_ax(
        inflow[inflow_sorting],
        ax=axs[0],
        color_mapper=tech_colors,
        )

    if not outflow.empty:
        print(outflow.columns)
        if "electricity demand" in outflow:
            outflow.rename(
                columns={
                    "electricity demand": "other electricity demand"
                    }, inplace=True
                )
        print(outflow.columns)

        outflow_sorting = (
            outflow[demands]
            .mean()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
       
        stackplot_to_ax(
            - outflow[outflow_sorting],
            ax=axs[1],
            color_mapper=tech_colors,
            )
    
    axs[2].plot(
        - outflow[outflow.columns.intersection(demands)].sum(axis=1),
        label="Demand",
    )

    axs[2].plot(
        inflow[inflow.columns.intersection(generators)].sum(axis=1),
        label="Generation",
    )

    shifter = pd.Index(shifters)
    charge = pd.DataFrame(index=inflow.index)

    if "PHS" in outflow.columns and "PHS" in inflow.columns:
        charge["hydro storage"] = (- outflow["PHS"] - inflow["PHS"]).cumsum()

    if "smart EV charger" in outflow.columns and "smart EV charger" in inflow.columns:
        charge["ev battery"] = (
            - outflow["smart EV charger"] * 0.9
            - inflow["smart EV charger"]
        )
        if "vehicle to grid" in inflow.columns:
            charge["ev battery"] -= inflow["vehicle to grid"] * 0.9
        
        charge["ev battery"] = charge["ev battery"].cumsum()

    if (
        "smart heat pump" in outflow.columns and 
        "smart heat pump" in inflow.columns):

        charge["residential building heat inertia"] = (
            - outflow["smart heat pump"]
            - inflow["smart heat pump"]).cumsum()

    if "grid battery" in outflow.columns and "grid battery" in inflow.columns:
        charge["grid battery"] = (
            - outflow["grid battery"]
            - inflow["grid battery"]
        ).cumsum()
    
    charge_plus = charge.clip(lower=0)
    charge_minus = charge.clip(upper=0)
    
    stackplot_to_ax(
        charge_plus,
        ax=axs[3],
        color_mapper=tech_colors,
        )
    stackplot_to_ax(
        charge_minus,
        ax=axs[3],
        color_mapper=tech_colors,
        legend=False,
        )

    axs[-1].set_ylabel("Volume Shifted Energy (GWh)")
    axs[-1].legend(loc="upper left")

    logger.warning("Kirchhoff currently switched off.")

    """
    if config["flexibility"]["timeseries_params"]["add_kirchhoff"]:
        axs.plot(total.index,
                total,
                color="black",
                label="Kirchhoff Check",
                linestyle="--",
                )
    """

    axs[0].set_ylabel("Generation (GW)")
    axs[1].set_ylabel("Load (GW)")
    axs[2].set_ylabel("Load vs Generation (GW)")

    axs[2].set_xlim(inflow.index[0], inflow.index[-1])
    for ax in axs:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        # switch all years to current year
        labels = [label.replace("2022", str(snakemake.wildcards.year)) for label in labels]
        ax.set_xticklabels(labels)

    title = (
        f"{config['flexibility']['timeseries_region'].upper()};"
        f" {scenario_mapper[snakemake.wildcards.fes]};"
        f" {snakemake.wildcards.year}"
    )
    # axs[0].set_title(title)
    axs[0].set_xticklabels([])

    # change x tick labels
    # labels = [item.get_text() for item in axs[1].get_xticklabels()]
    print(labels) 
    labels = ["Jan 1", "Jan 3", "Jan 5", "Jan 7", "Jan 9", "Jan 11", "Jan 13"]
    axs[1].set_xticklabels(labels)

    """
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
    """

    for ax in axs:
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(snakemake.output["timeseries"])
    plt.show()

    # make_co2_barplot(n)
