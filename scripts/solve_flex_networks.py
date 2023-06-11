# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors + Lukas Franken
#
# SPDX-License-Identifier: MIT

import logging

logger = logging.getLogger(__name__)

import os
import pypsa
import numpy as np

from _helpers import configure_logging
from solve_network import (override_component_attrs,
                           prepare_network,
                           solve_network)


if __name__ == "__main__":
    
    # this is basically a copy of solve_network.py 
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        
        snakemake = mock_snakemake(
            "solve_sector_network",
            configfiles="test/config.overnight.yaml",
            simpl="",
            opts="",
            clusters="5",
            ll="v1.5",
            sector_opts="CO2L0-24H-T-H-B-I-A-solar+p3-dist1",
            planning_horizons="2030",
        )
        
    configure_logging(snakemake)
    opts = snakemake.wildcards.opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.config["solving"]["options"]
    
    np.random.seed(solve_opts.get("seed", 123))

    if "overrides" in snakemake.input.keys():
        overrides = override_component_attrs(snakemake.input.overrides)
        n_base = pypsa.Network(snakemake.input["network"], override_component_attrs=overrides)
        n_flex = pypsa.Network(snakemake.input["network_flex"], override_component_attrs=overrides)
    else:
        n_base = pypsa.Network(snakemake.input["network"])
        n_flex = pypsa.Network(snakemake.input["network_flex"])

    for n, outfile in zip([n_base, n_flex],
                       [snakemake.output["network"],
                        snakemake.output["network_flex"]]):
        if os.path.isfile(outfile):
            logger.info(f"Skipping network \n {outfile} \n File already exists!")
            continue
            
        n = prepare_network(n, solve_opts, config=snakemake.config)

        n = solve_network(
            n, config=snakemake.config, opts=opts, log_fn=snakemake.log.solver
        )

        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(outfile)