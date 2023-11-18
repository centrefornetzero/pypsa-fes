#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script prepares generation and interconnector data from the ESPENI dataset
for a specific year.

It reads the ESPENI dataset, filters the data for a specific year, and resamples the
data on an hourly basis. The filtered and resampled generation data is saved to the
specified output file, while the filtered and resampled interconnector data is saved
to another output file.

Created in November 2023
    
"""

import logging

import pandas as pd

from _helpers import configure_logging

logger = logging.getLogger(__name__)


def get_beis_generation(file, year):

    def clean_string(strings):
        return [
            ''
            .join(filter(str.isalpha, s))
            .lower()
            .replace('note', '')
            for s in strings
            ]

    logger.info("Gathering BEIS generation (supplied) data.")
    df = pd.read_excel(
        file,
        sheet_name='5.6',
        index_col=1,
        header=161,
    ).iloc[:40]

    df = df.loc[df[df.columns[0]] == 'All generating companies']
    df = df.loc[df.index.dropna(), year]

    df.index = clean_string(df.index)
    df = df.groupby(df.index).sum().mul(1e-3)

    carriers = ['gas', 'solar', 'wind', 'coal', 'hydro', 'biomass', 'nuclear', 'oil', 'total']

    beis_clean = pd.Series(0, index=carriers, name='beis')
    beis_clean.loc['gas'] = df.loc['gas']

    beis_clean.loc[beis_clean.index.intersection(df.index)] = (
        df.loc[beis_clean.index.intersection(df.index)]
    )

    beis_clean.loc['wind'] = df.loc['onshorewind'] + df.loc['offshorewind']
    beis_clean.loc['biomass'] = df.loc['thermalrenewables']
    beis_clean.loc['hydro'] = df.loc['pumpedstorage'] + df.loc['hydronaturalflow']

    beis_clean.loc['total'] = df.loc["totalallgeneratingcompanies"]

    return beis_clean


if __name__ == '__main__':
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_network", simpl="", clusters="40", ll="v0.3", opts="Co2L-24H"
        )
    
    logger.info("Preparing generation and interconnector data from the ESPENI dataset.")
    configure_logging(snakemake)

    generator_mapper = {
        'POWER_ELEXM_CCGT_MW': 'CCGT',
        'POWER_ELEXM_OCGT_MW': 'OCGT',
        'POWER_ELEXM_OIL_MW': 'oil',
        'POWER_ELEXM_COAL_MW': 'coal',
        'POWER_ELEXM_NUCLEAR_MW': 'nuclear',
        'POWER_ELEXM_WIND_MW': 'wind',
        'POWER_ELEXM_NPSHYD_MW': 'NPSHYD',
        'POWER_ELEXM_BIOMASS_POSTCALC_MW': 'biomass',
        'POWER_ELEXM_OTHER_POSTCALC_MW': 'other',
        'POWER_NGEM_EMBEDDED_SOLAR_GENERATION_MW': 'solar',
        'POWER_NGEM_EMBEDDED_WIND_GENERATION_MW': 'embedded wind',
    }

    intercon_mapper = {
        'POWER_NGEM_BRITNED_FLOW_MW': 'BritNed',
        'POWER_NGEM_EAST_WEST_FLOW_MW': 'EAST_WEST',
        'POWER_NGEM_FRENCH_FLOW_MW': 'french flow',
        'POWER_NGEM_MOYLE_FLOW_MW': 'Moyle',
    }

    espeni = pd.read_csv(snakemake.input.espeni_dataset, index_col=2, parse_dates=True)

    year = '2019'

    generation = (
        espeni
        .loc[year, list(generator_mapper)]
        .rename(columns=generator_mapper)
        .resample('H').mean()
    )

    interconnectors = (
        espeni
        .loc[year, list(intercon_mapper)]
        .rename(columns=intercon_mapper)
        .resample('H').mean()
    )

    if snakemake.params['elec_config']['include_beis']:
        beis_generation = get_beis_generation(snakemake.input.beis_generation, year)
        beis_generation.to_csv(snakemake.output.beis_generation)

    generation.to_csv(snakemake.output.generation)
    interconnectors.to_csv(snakemake.output.interconnectors)
