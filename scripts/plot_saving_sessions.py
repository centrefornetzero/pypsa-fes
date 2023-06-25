# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Provides tools to analyse the turndown achieved in Saving Sessions through the generation
proposed by a PyPSA network.


Inputs
------
- ``data/ss_turndown.csv``: Turndown achieved during Saving Sessions 2022/23

Data Sources
------------
    [1] Achieved turndown
    internal source

"""

from copy import deepcopy
import pypsa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from _helpers import configure_logging
from plot_gb_validation import (
    preprocess_generation_pypsa,
    stackplot_to_ax,
    pypsa_mapper,
    tech_colors,)


def plot_saving_sessions_naive(n, ss_df):

    n = deepcopy(n)

    ss_df = deepcopy(ss_df).loc["2022"]
    gb_gen = n.generators.loc[n.generators.bus.str.contains("GB")]
    gb_gen = gb_gen.loc[gb_gen.carrier.isin(["CCGT", "OCGT", "coal", "lignite"])]

    fuel_savings = pd.DataFrame(index=ss_df.index,
                                columns=["fuel_type", "fuel_amount", "emissions"])    

    for time in ss_df.index:

        tslice = n.generators_t.p.loc[time, gb_gen.index]
        tslice = tslice.loc[tslice < gb_gen.p_nom]
        
        for gen in tslice.index:
            if not gen in n.generators_t.p_max_pu.columns:
                continue

            if not tslice.loc[gen] < n.generators_t.p_max_pu.at[time, gen] * gb_gen.at[gen, "p_nom"]:
                tslice.drop(gen, inplace=True) 

        mcosts = gb_gen.loc[tslice.index, "marginal_cost"] / gb_gen.loc[tslice.index, "efficiency"]

        mgen = mcosts.sort_values().index[0]

        fuel_savings.at[time, "fuel_type"] = gb_gen.loc[mgen, "carrier"]
        fuel_savings.at[time, "fuel_amount"] = ss_df.loc[time] / gb_gen.loc[mgen, "efficiency"] * 1e-3
        fuel_savings.at[time, "emissions"] = n.carriers.loc[gb_gen.loc[mgen, "carrier"], "co2_emissions"] * ss_df.loc[time] * 1e-3

    fuel_savings.index = [time.round("d") for time in fuel_savings.index]
    fuel_savings.index = ['2022-11-15', '2022-11-22', '2022-11-30', '2022-12-01',
               '2022-12-12']

    fuel_savings["dt"] = fuel_savings.index

    barplot_kwargs = {
        "edgecolor": "k",
        "alpha": 0.8,
        "width": 0.7,
        "color": "blue"
    }

    _, ax = plt.subplots(1, 1, figsize=(6, 3))
    sns.barplot(fuel_savings, x="dt", y="fuel_amount", ax=ax, **barplot_kwargs)
    ax.set_ylabel("Fuel Saved (MWh)")
    ax.set_xlabel("Event Date")

    plt.tight_layout()
    plt.savefig(snakemake.output["saved_fuel"])

    _, ax = plt.subplots(1, 1, figsize=(6, 3))
    sns.barplot(fuel_savings, x="dt", y="emissions", ax=ax, **barplot_kwargs)
    ax.set_ylabel("Emissions Saved (tCO2)")
    ax.set_xlabel("Event Date")

    plt.tight_layout()
    plt.savefig(snakemake.output["saved_emissions"])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_monthly_prices")

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input["network"])
    ss_df = pd.read_csv(snakemake.input["saving_sessions_data"], parse_dates=True, index_col=0)
    ss_df =ss_df["avgp376unclipped"]

    plot_saving_sessions_naive(n, ss_df)

    # plot Saving Sessions in a generation timeseries
    gen = preprocess_generation_pypsa(n, pypsa_mapper).resample("d").mean() * 1e-3

    _, ax = plt.subplots(1, 1, figsize=(16, 4))

    stackplot_to_ax(gen, ax, color_mapper=tech_colors)

    ax.set_ylabel("Generation (GW)")
    ax.set_xlabel("Datetime")

    l_kwargs = {"linewidth": 1, "alpha": 0.8, "color":"k"}

    for i, date in enumerate(ss_df.index):
        if not i:
            ax.axvline(date, label="Saving Session", **l_kwargs)
        else:
            ax.axvline(date, **l_kwargs)

    plt.tight_layout()
    plt.savefig(snakemake.output.generation_timeseries)