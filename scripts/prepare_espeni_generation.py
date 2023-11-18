#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script prepares generation and interconnector data from the ESPENI dataset for a specific year.

It reads the ESPENI dataset, filters the data for a specific year, and resamples the data on an hourly basis.
The filtered and resampled generation data is saved to the specified output file, while the filtered and resampled
interconnector data is saved to another output file.

Created in November 2023
    
"""

import logging

import pandas as pd

from _helpers import configure_logging

logger = logging.getLogger(__name__)

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
        'POWER_ELEXM_OIL_MW': 'OIL',
        'POWER_ELEXM_COAL_MW': 'COAL',
        'POWER_ELEXM_NUCLEAR_MW': 'NUCLEAR',
        'POWER_ELEXM_WIND_MW': 'WIND',
        'POWER_ELEXM_NPSHYD_MW': 'NPSHYD',
        'POWER_ELEXM_BIOMASS_POSTCALC_MW': 'BIOMASS',
        'POWER_ELEXM_OTHER_POSTCALC_MW': 'OTHER',
        'POWER_NGEM_EMBEDDED_SOLAR_GENERATION_MW': 'SOLAR',
        'POWER_NGEM_EMBEDDED_WIND_GENERATION_MW': 'EMBEDDED_WIND',
    }

    intercon_mapper = {
        'POWER_NGEM_BRITNED_FLOW_MW': 'BRITNED',
        'POWER_NGEM_EAST_WEST_FLOW_MW': 'EAST_WEST',
        'POWER_NGEM_FRENCH_FLOW_MW': 'FRENCH_FLOW',
        'POWER_NGEM_MOYLE_FLOW_MW': 'MOYLE',
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

    generation.to_csv(snakemake.output.generation)
    interconnectors.to_csv(snakemake.output.interconnectors)




