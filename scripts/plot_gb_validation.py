# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Creates plots of model optimisation results in GB

Validates against validation data from national grid ESO

Inputs
------
- ``data/gb_generation_2022.csv``: Generation profiles of GB from national grid ESO
    

Data Sources
------------
    [1] GB generation profiles; 'historic generation mix'
    https://data.nationalgrideso.com/carbon-intensity1/historic-generation-mix/r/historic_gb_generation_mix

"""

import logging

logger = logging.getLogger(__name__)

import pypsa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from copy import deepcopy


pypsa_mapper = {
    "nuclear": ["nuclear"],
    "biomass": ["biomass"],
    "hydro": ["ror"],
    "solar": ["solar"],
    #"oil": ["oil"],
    "wind": ["offwind-ac", "offwind-dc", "onwind"],
    "gas": ["CCGT", "OCGT"],
    "coal": ["coal", "lignite"]
}

eso_mapper = {key: key.upper() for key in list(pypsa_mapper)}


def preprocess_generation_pypsa(n, mapper, country="GB"):

    gb_generation = n.generators.loc[n.generators.bus.str.contains(country)]
    
    ts = pd.DataFrame(index=n.generators_t.p.index)

    for tech, carriers in mapper.items():
        gens = gb_generation.loc[gb_generation.carrier.isin(carriers)].index
        ts[tech] = n.generators_t.p[gens].sum(axis=1)

    return ts


def get_transmission_pypsa(n, country="GB"):

    mask1 = n.links.bus0.str.startswith(country)
    mask2 = n.links.bus1.str.startswith(country)

    intercon = n.links.loc[mask1 ^ mask2]

    gb0_flow = intercon.loc[intercon.bus0.str.startswith(country)].index
    gb1_flow = intercon.drop(gb0_flow).index

    intercon_flow = pd.concat((
        n.links_t.p1[gb0_flow], n.links_t.p0[gb1_flow]
    ), axis=1)

    intercon_flow_positive = pd.DataFrame({"transmission lines":
        intercon_flow.sum(axis=1).clip(lower=0.)})

    intercon_flow_negative = pd.DataFrame({"transmission lines":
        intercon_flow.sum(axis=1).clip(upper=0.)})

    return intercon_flow_positive, intercon_flow_negative


def get_storage_flows_pypsa(n, country="GB"):
    storage = n.storage_units.loc[
        n.storage_units.bus.str.startswith(country)].index

    sp = pd.DataFrame({"ror": n.storage_units_t.p[storage].sum(axis=1)})

    storage_dispatch = sp.clip(lower=0.)
    storage_charging = sp.clip(upper=0.)

    return storage_dispatch, storage_charging


def get_store_flows_pypsa(n, country="GB"):

    stores = n.stores.loc[n.stores.bus.str.startswith(country)]

    df_charge = pd.DataFrame(index=n.links_t.p0.index)
    df_discharge = pd.DataFrame(index=n.links_t.p0.index)

    for carrier in stores.carrier.unique():

        stores = stores.loc[stores.carrier == carrier]

        store_chargers = n.links.loc[n.links.bus1.isin(stores.bus)]
        df_charge[carrier+" charge"] = (
            n.links_t
            .p1[store_chargers.index]
            # .mul(n.links.loc[store_chargers.index].efficiency, axis=1)     
        ).sum(axis=1)

        store_dischargers = n.links.loc[n.links.bus0.isin(stores.bus)]
        discharge = (
            n.links_t
            .p0[store_dischargers.index]
            .mul(n.links.loc[store_dischargers.index].efficiency, axis=1)     
        )

        df_discharge[carrier+" dispatch"] = discharge.sum(axis=1)

    return df_discharge, df_charge


def get_load_pypsa(n, country="GB"):

    gb_buses = n.loads.loc[n.loads.bus.str.startswith(country)].index
    load = n.loads_t.p_set[gb_buses].sum(axis=1)
    return load


def preprocess_generation_eso(df, mapper):

    ts = pd.DataFrame(index=df.index)
    for target, origin in mapper.items():
        ts[target] = df[origin]
    return ts


def stackplot_to_ax(df, ax, color_mapper={}, stackplot_kwargs={}):
    if color_mapper:
        colors = [color_mapper[tech] for tech in df.columns]
    else:
        colors = None

    stackplot_kwargs_default = {
        "edgecolor": "k", 
        "linewidth": 0.1,
        "linestyle": ":",
        "alpha": 0.9,
        }
    stackplot_kwargs_default.update(stackplot_kwargs)
    
    ax.stackplot(df.index,
                 df.values.T,
                 colors=colors,
                 labels=df.columns,
                 **stackplot_kwargs_default)
    ax.set_xlim(df.index[0], df.index[-1])


def compare_generation_timeseries(
    gen_real,
    gen_model_inflow,
    gen_model_outflow,
    load_model=None,
    start=None,
    end=None,
    freq=None,
    vlines=None,
    savefile=None):

    if load_model is None:
        kirchhoff = pd.DataFrame(index=gen_real.index)
    else:
        if load_model.index.tz is None:
            load_model.index = load_model.index.tz_localize("UTC")
        kirchhoff = load_model - gen_model_inflow.sum(axis=1) - gen_model_outflow.sum(axis=1) 

    tech_colors = snakemake.config["plotting"]["tech_colors"]
    tech_colors["wind"] = tech_colors["onwind"]
    tech_colors["H2 dispatch"] = tech_colors["H2 Fuel Cell"]
    tech_colors["H2 charge"] = tech_colors["H2 Electrolysis"]
    tech_colors["battery dispatch"] = tech_colors["battery"]
    tech_colors["battery charge"] = tech_colors["BEV charger"]
    
    assert (int(bool(start)) + int(bool(end))) % 2 == 0, "Please choose either both or none of start, end"
    
    if start is not None:
        gen_real = gen_real.loc[start:end]
        gen_model_inflow = gen_model_inflow.loc[start:end]
        gen_model_outflow = gen_model_outflow.loc[start:end]
        kirchhoff = kirchhoff.loc[start:end]

    if freq is not None:
        gen_real = gen_real.resample(freq).mean()
        gen_model_inflow = gen_model_inflow.resample(freq).mean()
        gen_model_outflow = gen_model_outflow.resample(freq).mean()
        kirchhoff = kirchhoff.resample(freq).mean()

    _, axs = plt.subplots(1, 2, figsize=(16, 4))

    stackplot_to_ax(gen_real, axs[0], color_mapper=tech_colors)
    stackplot_to_ax(gen_model_inflow, axs[1], color_mapper=tech_colors)
    stackplot_to_ax(gen_model_outflow, axs[1], color_mapper=tech_colors)
        
    if not kirchhoff.empty:
        axs[1].plot(kirchhoff.index,
                    kirchhoff.values,
                    color="k",
                    linewidth=1.2,
                    label="Kirchhoff balance",
                    )

    axs[0].set_ylabel("Generation (GW)")

    for ax, title in zip(axs, ["ESO data", "PyPSA"]):
        ax.set_xlabel("Datetime")
        ax.set_title(title)

    if vlines is not None:

        l_kwargs = {"linewidth": 1, "alpha": 0.8, "color":"k"}
        for ax in axs:
            for i, date in enumerate(vlines.index):
                if not i:
                    ax.axvline(date, label="Saving Session", **l_kwargs)
                else:
                    ax.axvline(date, **l_kwargs)

    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)

    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()


def compare_totals(gen_eso, gen_pypsa, savefile=None):
    
    gen_eso = deepcopy(gen_eso)
    gen_pypsa = deepcopy(gen_pypsa)

    eso_totals = pd.DataFrame(gen_eso.sum().values, index=gen_eso.columns, columns=["tech"]) * 1e-3
    pypsa_totals = pd.DataFrame(gen_pypsa.sum().values, index=gen_pypsa.columns, columns=["tech"]) * 1e-3

    eso_totals["source"] = ["eso" for _ in range(len(eso_totals))]
    pypsa_totals["source"] = ["pypsa" for _ in range(len(pypsa_totals))]

    eso_totals["carrier"] = eso_totals.index
    pypsa_totals["carrier"] = pypsa_totals.index

    totals = pd.concat((eso_totals, pypsa_totals), axis=0)

    _, ax = plt.subplots(1, 1, figsize=(8, 5))

    sns.barplot(data=totals,
                x="carrier",
                y="tech",
                hue="source",
                ax=ax,
                edgecolor="k",
                alpha=0.9,
                )

    ax.set_xlabel("Technology")
    ax.set_ylabel("Total Generation (TWh)")

    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)



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

    n = pypsa.Network(snakemake.input["network"])
    gen = pd.read_csv(snakemake.input["gb_generation_data"],
                      parse_dates=True,
                      index_col=0)

    gen_pypsa = preprocess_generation_pypsa(n, pypsa_mapper)
    loads_pypsa = get_load_pypsa(n) * 1e-3

    storage_plus, storage_minus = get_storage_flows_pypsa(n)
    store_plus, store_minus = get_store_flows_pypsa(n)
    transmission_plus, transmission_minus = get_transmission_pypsa(n)

    energy_inflows = pd.concat((
        gen_pypsa, storage_plus, store_plus, transmission_plus
    ), axis=1) * 1e-3
    energy_outflows = pd.concat((
        storage_minus, store_minus, transmission_minus
    ), axis=1) * 1e-3
    gen_pypsa *= 1e-3

    gen_eso = preprocess_generation_eso(gen, eso_mapper) * 1e-3
    gen_pypsa.index = gen_eso.index
    energy_inflows.index = gen_eso.index
    energy_outflows.index = gen_eso.index

    compare_generation_timeseries(
        gen_eso,
        energy_inflows,
        energy_outflows,
        load_model=loads_pypsa,
        freq="d",
        savefile=snakemake.output["generation_timeseries_year"]
        )
    compare_generation_timeseries(
        gen_eso,
        energy_inflows,
        energy_outflows, 
        load_model=loads_pypsa,
        start=pd.Timestamp("2022-11-13"),
        end=pd.Timestamp("2022-12-15"),
        savefile=snakemake.output["generation_timeseries_weeks"]
        )
    
    compare_totals(gen_eso, gen_pypsa, savefile=snakemake.output.total_generation)
