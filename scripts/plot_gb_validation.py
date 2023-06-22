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
import matplotlib.pyplot as plt
plt.style.use("ggplot")

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

tech_colors = {
    "nuclear": '#ff8c00',
    "solar": "#f9d002",
    "wind": "#6895dd",
    "hydro": '#298c81',
    "solar": "#f9d002",
    "gas": '#e05b09',
    "coal": '#545454',
    "oil": '#c9c9c9',
    "biomass": '#baa741',
}


def preprocess_generation_pypsa(n, mapper, country="GB"):

    gb_generation = n.generators.loc[n.generators.bus.str.contains(country)]
    
    ts = pd.DataFrame(index=n.generators_t.p.index)

    for tech, carriers in mapper.items():
        gens = gb_generation.loc[gb_generation.carrier.isin(carriers)].index
        ts[tech] = n.generators_t.p[gens].sum(axis=1)

    return ts


def preprocess_generation_eso(df, mapper):

    ts = pd.DataFrame(index=df.index)
    for target, origin in mapper.items():
        ts[target] = df[origin]
    return ts


def stackplot_to_ax(df, ax, color_mapper={}):
    if color_mapper:
        colors = [color_mapper[tech] for tech in df.columns]
    else:
        colors = None
    
    ax.stackplot(df.index,
                 df.values.T,
                 colors=colors,
                 labels=df.columns)
    ax.set_xlim(df.index[0], df.index[-1])

def compare_generation_timeseries(
    gen_real,
    gen_model,
    start=None,
    end=None,
    freq=None,
    vlines=None,
    savefile=None):
    
    assert (int(bool(start)) + int(bool(end))) % 2 == 0, "Please choose either both or none of start, end"
    
    if start is not None:
        gen_real = gen_real.loc[start:end]
        gen_model = gen_model.loc[start:end]
    
    if freq is not None:
        gen_real = gen_real.resample(freq).mean()
        gen_model = gen_model.resample(freq).mean()
    
    _, axs = plt.subplots(1, 2, figsize=(16, 4))
    
    stackplot_to_ax(gen_real, axs[0], color_mapper=tech_colors)
    stackplot_to_ax(gen_model, axs[1], color_mapper=tech_colors)

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
    
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
    
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()
    





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
    gen = pd.read_csv(snakemake.input["gb_generation_data"], parse_dates=True, index_col=0)

    gen_pypsa = preprocess_generation_pypsa(n, pypsa_mapper)
    gen_eso = preprocess_generation_eso(gen, eso_mapper)
    gen_pypsa.index = gen_eso.index

    compare_generation_timeseries(gen_eso,
        gen_pypsa, 
        freq="d",
        savefile=snakemake.output["generation_timeseries_year"]
        )
    compare_generation_timeseries(gen_eso,
        gen_pypsa,
        start=pd.Timestamp("2022-11-13"),
        end=pd.Timestamp("2022-12-15"),
        savefile=snakemake.output["generation_timeseries_weeks"]
        )
    
    raise NotImplementedError("Add saving sessions and total generation plot")
    



    