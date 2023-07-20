# SPDX-FileCopyrightText: : 2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


localrules:
    all,
    cluster_networks,
    extra_components_networks,
    prepare_elec_networks,
    # prepare_sector_networks,
    solve_elec_networks,
    # solve_sector_networks,
    plot_networks,
    # plot_gb_validation,


rule all:
    input:
        RESULTS + "graphs/costs.pdf",
    default_target: True


rule cluster_networks:
    input:
        expand(RESOURCES + "networks/elec_s{simpl}_{gb_regions}.nc", **config["scenario"]),


rule extra_components_networks:
    input:
        expand(
            RESOURCES + "networks/elec_s{simpl}_{gb_regions}_ec.nc", **config["scenario"]
        ),


rule prepare_elec_networks:
    input:
        expand(
            RESOURCES + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{fes_scenario}_{planning_horizons}.nc",
            **config["scenario"]
        ),


rule prepare_sector_networks:
    input:
        expand(
            RESULTS
            + "prenetworks/elec_s{simpl}_{gb_regions}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"]
        ),


rule solve_elec_networks:
    input:
        expand(
            RESULTS + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{fes_scenario}_{planning_horizons}.nc",

            **config["scenario"]
        ),


rule solve_sector_networks:
    input:
        expand(
            RESULTS
            + "postnetworks/elec_s{simpl}_{gb_regions}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"]
        ),


rule plot_networks:
    input:
        expand(
            RESULTS
            + "maps/elec_s{simpl}_{gb_regions}_l{ll}_{opts}_{sector_opts}-costs-all_{fes_scenario}_{planning_horizons}.pdf",
            **config["scenario"]
        ),