#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue June 28 2023; built by Lukas Franken

Builds constraints for generation capacities based on the FES scenarios

- capacity_constraint: pd.DataFrame columns carrier, attr, value, sense
    (further processed in solve_network.py)
- load_constraint: pd.DataFrame with cols tech, value
- battery_constraints: pd.DataFrame with cols component, attr, value, sense

"""

import pandas as pd
import numpy as np

from _fes_helpers import (
    get_data_point,
    get_interconnector_capacity
    )
from _helpers import configure_logging


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_fes_scenarios")

    configure_logging(snakemake)

    fes = snakemake.wildcards["fes"]
    year = int(snakemake.wildcards["year"])

    caps = pd.DataFrame(columns=["carrier", "attr", "value", "sense"]) 
    loads = pd.DataFrame(columns=["carrier", "attr", "value", "sense"]) 

    val = get_data_point("onshore_wind_capacity", fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "onwind",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})

    val = get_data_point("offshore_wind_capacity", fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "offwind",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})

    val = get_data_point("solar_capacity", fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "solar",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})

    # (unabated gas)
    val = get_data_point("gas_capacity", fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "gas",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})
    
    """
    val = get_data_point("gas_ccs_capacity", fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "gas ccs",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})
    """
    
    val = get_data_point("nuclear_capacity", fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "nuclear",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})

    coal_phaseout = snakemake.config["flexibility"]["coal_phaseout_year_uk"]
    caps.loc[len(caps)] = pd.Series({
        "carrier": "coal",
        "attr": "p_nom",
        "value": np.interp(year, [2023, coal_phaseout], [2_520., 0]),
        "sense": "==",})

    val = get_interconnector_capacity(fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "DC",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})

    """
    val = get_data_point("bioenergy_capacity", fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "biomass",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})

    val = get_data_point("bioenergy_ccs_capacity", fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "biomass ccs",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})
    """

    caps.to_csv(snakemake.output["capacity_constraints"])
