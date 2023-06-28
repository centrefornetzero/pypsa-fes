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

Description
-----------

Data sources:
    [1] Future energy scenarios workbook
    https://www.nationalgrideso.com/document/263876/download


"""

import pandas as pd
import numpy as np
import string

data_file = "../data/Data-workbook2022_V006.xlsx"

page_mapper = {
    "wind_capacity": {"sheet": "ES.E.12", "start_col": "N", "start_row": 9, "unit": "GW", "format": "gen"},
    "offshore_wind_capacity": {"sheet": "ES.E.13", "start_col": "N", "start_row": 8, "unit": "GW", "format": "gen"},
    "onshore_wind_capacity": {"sheet": "ES.E.14", "start_col": "N", "start_row": 8, "unit": "GW", "format": "gen"},
    "solar_capacity": {"sheet": "ES.E.16", "start_col": "M", "start_row": 7, "unit": "GW", "format": "gen"},
    "gas_capacity": {"sheet": "ES.E.17", "start_col": "I", "start_row": 7, "unit": "GW", "format": "gen"},
    "gas_ccs_capacity": {"sheet": "ES.E.18", "start_col": "I", "start_row": 7, "unit": "GW", "format": "gen"},
    "bioenergy_ccs_capacity": {"sheet": "ES.E.19", "start_col": "I", "start_row": 7, "unit": "GW", "format": "gen"},
    "bioenergy_capacity": {"sheet": "ES.E.20", "start_col": "I", "start_row": 7, "unit": "GW", "format": "gen"},
    "nuclear_capacity": {"sheet": "ES.E.21", "start_col": "M", "start_row": 7, "unit": "GW", "format": "gen"},
    "battery_charge_capacity": {"sheet": "ES.E.26", "start_col": "M", "start_row": 7, "unit": "GW", "format": "gen"},
    "battery_discharge_capacity": {"sheet": "ES.E.26", "start_col": "M", "start_row": 7, "unit": "GW", "format": "gen"},

    "ashp_installations": {"sheet": "EC.R.10", "start_col": "M", "start_row": 9, "unit": "#", "format": "cumul"},
    "elec_demand_home_heating": {"sheet": "EC.R.06", "start_col": "M", "start_row": 8, "unit": "GWh", "format": "gen"},
    "white_good_response": {"sheet": "EC.R.16", "start_col": "M", "start_row": 8, "unit": "%", "format": "gen"},
    
}

scenario_mapper = {
    "CT": "Consumer Transformation", 
    "ST": "System Transformation", 
    "LW": "Leading the Way", 
    "FS": "Falling Short",
    }

extra = ["elec_demand_home_heating"]

def get_extra(datapoint, scenario, year):
    """Extracts datapoints that do not confirm to a common format, and need more specialized
    functions to obtain"""

    sheet, col, row, unit, format = list(page_mapper[datapoint].values())
    col = string.ascii_uppercase.index(col)

    if datapoint == "elec_demand_home_heating":
        
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
        print(df)

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
        df.columns = [pd.Timestamp(dt).year for dt in df.columns]
        val = df.loc[scenario_mapper[scenario], year]
        return val * 1e3

    elif format == "cumul":
        val = df.loc[scenario_mapper[scenario], :year].sum()
        return val


get_data_point("white_good_response", "FS", 2021)