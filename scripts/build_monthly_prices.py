#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:37:35 2023; built by Lisa Zeyen in
branch `validation` of PyPSA-Eur; extended by Lukas Franken

Extracts monthly prices of fuel and CO2 emissions for both mainland Europe,
and Great Britain.
Considered fuels are oil, gas, lignite, and coal

Inputs
------
- ``data/energy-price-trends-xlsx-5619002.xlsx``: energy price index of fossil fuels
- ``emission-spot-primary-market-auction-report-2019-data.xls``: CO2 Prices spot primary auction


Outputs
-------

- ``data/monthly_fuel_prices.csv``
- ``data/CO2_prices_2022.csv``

Description
-----------

The rule :mod:`build_monthly_prices` collects monthly fuel prices and CO2 prices
and translates them from different input sources to pypsa syntax

Data sources:
    [1] Fuel price index. Destatis
    https://www.destatis.de/EN/Home/_node.html
    [2] average annual import price (coal, gas, oil) Agora, slide 24
    https://static.agora-energiewende.de/fileadmin/Projekte/2019/Jahresauswertung_2019/A-EW_German-Power-Market-2019_Summary_EN.pdf
    [3] average annual fuel price lignite, ENTSO-E
    https://2020.entsos-tyndp-scenarios.eu/fuel-commodities-and-carbon-prices/
    [4] Mainland Europe CO2 Prices, Emission spot primary auction, EEX
    https://www.eex.com/en/market-data/environmental-markets/eua-primary-auction-spot-download
    [5] Yearly UK CO2 Prices, UK government,  
    https://www.gov.uk/government/publications/determinations-of-the-uk-ets-carbon-price/uk-ets-carbon-prices-for-use-in-civil-penalties-2021-and-2022
    https://www.gov.uk/government/publications/determinations-of-the-uk-ets-carbon-price/uk-ets-carbon-prices-for-use-in-civil-penalties-2023
    [6] UK fuel prices,
    https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1146301/table_321.xlsx


"""

import pandas as pd
import numpy as np
import logging
from _helpers import configure_logging

logger = logging.getLogger(__name__)

model_year = 2022

# sheet names to pypsa syntax
sheet_name_map = {"5.1 Hard coal and lignite": "coal",
               "5.2 Mineral oil" : "oil",
               "5.3.1 Natural gas - indices":"gas"}

# keywords in datasheet
keywords = {"coal": " GP09-051 Hard coal",
            "lignite": " GP09-052 Lignite and lignite briquettes",
            "oil": " GP09-0610 10 Mineral oil, crude",
            "gas": "GP09-062 Natural gas"
            }


# Great Britain CO2 price from [5]
co2_prices_UK = {
    2021: 47.96, # £/tCO2
    2022: 52.56, # £/tCO2
    2023: 83.03, # £/tCO2
}

price_2015 = {"coal": 8.3,
              "oil": 30.6,
              "gas": 20.6,
              "lignite": 3.8}  # 2020 3.96/1.04 


def get_fuel_price():
    mainland_prices = pd.read_excel(
        snakemake.input["eu_fuel_price_raw"],
        index_col=1,
        header=5)
    pass


def get_co2_price():

    eu_co2_price = pd.read_excel(snakemake.input.eu_co2_price_raw,
                              index_col=1,
                              header=5,
                              )

    exchange_rate = pd.read_csv(
        snakemake.input["exchange_rate"],
        index_col=0,
        header=16,
        parse_dates=True,
        )

    eu_prices = eu_co2_price["Auction Price €/tCO2"].values
    index = eu_co2_price["Auction Price €/tCO2"].index
    exchange_rate = exchange_rate.loc[index].values.flatten()

    eu_prices *= exchange_rate # make prices Pound/MWh

    countries = snakemake.config["countries"]
    countries.remove("GB") 

    co2prices = pd.DataFrame(
        np.stack((eu_prices for _ in range(len(countries)))).T,
        columns=list(countries),
        index=index,
    ).iloc[::-1].interpolate()

    co2prices["GB"] = [co2_prices_UK[time.year] for time in co2prices.index]

    assert not co2prices.isna().any().any(), "Detected nan values in CO2 prices"

    co2prices.to_csv(snakemake.output["co2_price"])


def get_fuel_price():

    logger.warning("Currently assuming uniform fuel prices across Europe.")
    
    exchange_rate = pd.read_csv(
        snakemake.input["exchange_rate"],
        index_col=0,
        header=16,
        parse_dates=True,
        )
    rate = exchange_rate.loc["2022"].resample("m").mean().values.flatten()

    fuel_price = pd.read_excel(snakemake.input["eu_fuel_price_raw"],
                sheet_name=list(sheet_name_map.keys()))
    fuel_price = {sheet_name_map[key]: value for key, value in fuel_price.items()
        if key in sheet_name_map}
    # lignite and hard coal are on the same sheet
    fuel_price["lignite"] = fuel_price["coal"]

    def extract_df(sheet, keyword):
        # Create a DatetimeIndex for the first day of each month of a given year
        dti = pd.date_range(start=f'{model_year}-01-01',
                            end=f'{model_year}-12-01', freq='MS')
        # Extract month names
        month_list = dti.month
        start = fuel_price[sheet].index[(fuel_price[sheet] == keyword).any(axis=1)]
        df = fuel_price[sheet].loc[start[0]:start[0]+18,:]
        df.dropna(axis=0, inplace=True)
        df.iloc[:,0] = df.iloc[:,0].apply(lambda x: int(x.replace(" ...", "")))
        df.set_index(df.columns[0], inplace=True)
        df = df.iloc[:, :12]
        df.columns = month_list
        return df

    m_price = {}

    for carrier, keyword in keywords.items():
        df = extract_df(carrier, keyword).loc[model_year]
        m_price[carrier] = df.mul(price_2015[carrier]/100)

    m_price = pd.concat(m_price, axis=1)
    m_price = m_price.multiply(rate, axis=0)

    m_price.to_csv(snakemake.output["fuel_price"])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_monthly_prices")

    configure_logging(snakemake)

    get_co2_price()
    get_fuel_price()