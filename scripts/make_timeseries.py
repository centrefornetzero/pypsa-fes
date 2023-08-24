# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors, Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Creates timeseries of generation profiles for desired regions

"""

import logging

logger = logging.getLogger(__name__)

import pypsa
import numpy as np
import pandas as pd
from pprint import pprint

from make_summary import assign_carriers, assign_locations
from _helpers import override_component_attrs


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("make_timeseries")

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    assign_carriers(n)
    assign_locations(n)

    config = snakemake.config

    if  (region := config["flexibility"]["timeseries_region"]) != "gb":
        assert region in config["flexibility"]["regions"], f"Region {region} not in flexibility.regions"

        buses = config["flexibility"]["regions"][region]
    
    else:
        buses = pd.Index(n.buses.location.unique())
        buses = buses[buses.str.contains("GB")]

    load = n.loads_t.p_set.loc[:, n.loads.index.intersection(buses)].sum(axis=1)
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
    assert not pd.concat((inflow, outflow), axis=1).isna().any().any(), "NaNs in inflow or outflow."

    total_balance = abs((inflow.sum().sum() + outflow.sum().sum()) / inflow.sum().sum())
    if not np.allclose(total_balance, 0., atol=1e-3):
        logger.warning(f"Total imbalance in- and outflow {total_balance*100:.2f}% exceeds 0.1%.")

    inflow *= 1e-3
    outflow *= 1e-3

    logger.info(f"Created Timeseries inflow carriers: \n {', '.join(inflow.columns.tolist())}")
    logger.info(f"Created Timeseries outflow carriers: \n {', '.join(outflow.columns.tolist())}")

    inflow.to_csv(snakemake.output.inflow)
    outflow.to_csv(snakemake.output.outflow)