#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue June 28 2023; built by Lukas Franken

Extracts data from Future Energy Scenarios

Inputs
------
- ``data/Data-workbook2022_V006.xlsx`` FES data workbook

Outputs
-------

- capacity_constraint: pd.DataFrame columns carrier, attr, value, sense
    (further processed in solve_network.py)
- load_constraint: pd.DataFrame with cols tech, value
- battery_constraints: pd.DataFrame with cols component, attr, value, sense


Description
-----------

Data sources:
    [1] Future energy scenarios workbook
    https://www.nationalgrideso.com/document/263876/download


"""

import pandas as pd
import numpy as np
import string
from _helpers import configure_logging

data_file = "data/Data-workbook2022_V006.xlsx"

page_mapper = {
    "wind_capacity": {"sheet": "ES.E.12", "start_col": "N", "start_row": 9, "unit": "GW", "format": "gen"},
    "offshore_wind_capacity": {"sheet": "ES.E.13", "start_col": "N", "start_row": 8, "unit": "GW", "format": "gen"},
    "onshore_wind_capacity": {"sheet": "ES.E.14", "start_col": "N", "start_row": 8, "unit": "GW", "format": "gen"},
    "solar_capacity": {"sheet": "ES.E.16", "start_col": "M", "start_row": 7, "unit": "GW", "format": "gen"},
    "domestic_solar_capacity": {"sheet": "EC.R.19", "start_col": "N", "start_row": 6, "unit": "MW", "format": "gen"},
    "gas_capacity": {"sheet": "ES.E.17", "start_col": "I", "start_row": 7, "unit": "GW", "format": "gen"},
    "gas_ccs_capacity": {"sheet": "ES.E.18", "start_col": "I", "start_row": 7, "unit": "GW", "format": "gen"},
    "bioenergy_capacity": {"sheet": "ES.E.20", "start_col": "I", "start_row": 7, "unit": "GW", "format": "gen"},
    "bioenergy_ccs_capacity": {"sheet": "ES.E.19", "start_col": "I", "start_row": 7, "unit": "GW", "format": "gen"},
    "nuclear_capacity": {"sheet": "ES.E.21", "start_col": "M", "start_row": 7, "unit": "GW", "format": "gen"},
    "battery_charge_capacity": {"sheet": "ES.E.26", "start_col": "M", "start_row": 7, "unit": "GW", "format": "gen"},
    "battery_discharge_capacity": {"sheet": "ES.E.26", "start_col": "M", "start_row": 7, "unit": "GW", "format": "gen"},

    "ashp_installations": {"sheet": "EC.R.10", "start_col": "M", "start_row": 9, "unit": "#", "format": "cumul"},
    "elec_demand_home_heating": {"sheet": "EC.R.06", "start_col": "M", "start_row": 8, "unit": "GWh", "format": "gen"},
    "white_good_response": {"sheet": "EC.R.16", "start_col": "M", "start_row": 8, "unit": "%", "format": "gen"},
    
    "hydrogen_demand": {"sheet": "EC.R.08", "start_col": "M", "start_row": 7, "unit": "TWh", "format": "gen"}, # for home heating
    "total_emissions": {"sheet": "NZ.09"},
}

scenario_mapper = {
    "CT": "Consumer Transformation", 
    "ST": "System Transformation", 
    "LW": "Leading the Way", 
    "FS": "Falling Short",
    }

extra = ["elec_demand_home_heating", "total_emissions", "domestic_solar_capacity"]

def get_extra(datapoint, scenario, year):
    """Extracts datapoints that do not confirm to a common format, and need more specialized
    functions to obtain"""

    if datapoint == "domestic_solar_capacity":

        sheet, col, row, unit, format = list(page_mapper[datapoint].values())
        col = string.ascii_uppercase.index(col)

        df = (
            pd.read_excel(data_file,
                sheet_name=sheet,
                header=row,
                index_col=0,
                nrows=5,
                usecols=[col+i for i in range(37)],
                )
        ).iloc[1:]
        df.columns = [pd.Timestamp(dt).year for dt in df.columns]
        df = df.loc[:,2020:]

        return df.loc[scenario_mapper[scenario], year]

    if datapoint == "total_emissions":
        sheet = page_mapper[datapoint]["sheet"]
        scenarios = ["CT", "ST", "LW", "FS"]

        col = string.ascii_uppercase.index("M")
        df = (
            pd.read_excel(data_file,
                sheet_name=sheet,
                header=8,
                index_col=0,
                usecols=[col+i for i in range(32)],
                )
        )

        df = pd.concat((df.loc["Net1"], df.loc["Net"].transpose()), axis=1).transpose()
        df.index = scenarios

        if not (isinstance(df.columns[0], int) or isinstance(df.columns[0], np.int64)):
            df.columns = [pd.Timestamp(dt).year for dt in df.columns]

        return df.loc[scenario, year]


    if datapoint == "elec_demand_home_heating":

        sheet, col, row, unit, format = list(page_mapper[datapoint].values())
        col = string.ascii_uppercase.index(col)
        
        df = (
            pd.read_excel(data_file,
                sheet_name=sheet,
                header=row-2,
                index_col=0,
                nrows=5,
                usecols=[col+i for i in range(32)],
                )
            .iloc[1:]
        )
        df.columns = [pd.Timestamp(dt).year for dt in df.columns]
        df = df.loc[:,2020:]

        return df.loc[scenario_mapper[scenario], year] * 1e3


def get_data_point(datapoint, scenario, year):

    assert year >= 2020 and year <= 2050, f"Choose year between 2020 and 2050 instead of {year}."
    assert datapoint in page_mapper, (f"Datapoint {datapoint} is not implemented right now."
                                      f" Available are {list(page_mapper)}.")
    assert scenario in list(scenario_mapper), (f"Please choose one of {list(scenario_mapper)}"
                                               f" instead of {scenario}.")

    if datapoint in extra:
        return get_extra(datapoint, scenario, year)

    else:
        sheet, col, row, unit, format = list(page_mapper[datapoint].values())
        col = string.ascii_uppercase.index(col)

        df = (
            pd.read_excel(data_file,
                sheet_name=sheet,
                header=row-2,
                index_col=0,
                nrows=4,
                usecols=[col+i for i in range(32)],
                )
        )

    if format == "gen":
        if not (isinstance(df.columns[0], int) or isinstance(df.columns[0], np.int64)):
            df.columns = [pd.Timestamp(dt).year for dt in df.columns]

        val = df.loc[scenario_mapper[scenario], year]

        if unit == "GW":
            val = val * 1e3
        elif unit == "TWh":
            val = val * 1e6

    elif format == "cumul":
        val = df.loc[scenario_mapper[scenario], :year].sum()
        return val


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_fes_scenarios")

    configure_logging(snakemake)

    fes = snakemake.wildcards["fes_scenarios"]
    year = snakemake.wildcards["planning_horizons"]

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
        "carrier": "offwind-ac",
        "attr": "p_nom",
        "value": val/2.,
        "sense": "==",})
    caps.loc[len(caps)] = pd.Series({
        "carrier": "offwind-dc",
        "attr": "p_nom",
        "value": val/2.,
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
        "carrier": "OCGT",
        "attr": "p_nom",
        "value": val/2.,
        "sense": "==",})
    caps.loc[len(caps)] = pd.Series({
        "carrier": "CCGT",
        "attr": "p_nom",
        "value": val/2.,
        "sense": "==",})
    
    val = get_data_point("gas_ccs_capacity", fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "gas ccs",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})
    
    val = get_data_point("nuclear_capacity", fes, year)
    caps.loc[len(caps)] = pd.Series({
        "carrier": "nuclear",
        "attr": "p_nom",
        "value": val,
        "sense": "==",})

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
    
    caps.to_csv(snakemake.output["capacity_constraints"])
