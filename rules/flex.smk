# SPDX-FileCopyrightText: : 2023 Lukas Franken
#
# SPDX-License-Identifier: MIT

rule build_flex_network:
    input:
        network=RESOURCES + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.nc",
        flex_data="data/turndown_events/turndown_{td_event}.csv"
    output:
        network=RESOURCES + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_flex_event_{td_event}.nc",
    log:
        LOGS + "build_flex_network_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_flex_event_{td_event}.log",
    benchmark:
        BENCHMARKS + "build_flex_network_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_flex_event_{td_event}"
    threads: 1
    resources:
        mem_mb=4000,
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_flex_network.py"


rule solve_flex_networks:
    input:
        network=RESOURCES + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.nc",
        network_flex=RESOURCES + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_flex_event_{td_event}.nc",
    output:
        network=RESULTS + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_base_event_{td_event}.nc",
        network_flex=RESULTS + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_flex_event_{td_event}.nc",
    log:
        solver=normpath(
            LOGS + "solve_flex_network/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_flex_event_{td_event}_solver.log"
        ),
        python=LOGS
        + "solve_flex_network/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_flex_event_{td_event}_python.log",
    benchmark:
        BENCHMARKS + "solve_flex_networks/solve_flex_networks_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_flex_event_{td_event}"
    threads: 4
    resources:
        mem_mb=memory,
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_flex_networks.py"


rule make_gb_summary:
    input:
        regions_onshore=RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
        network=RESULTS + "networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
    output:
        regions_map=RESULTS + "plots/network_regions_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        capacity_expansion=RESULTS + "plots/capacity_expansion_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        generation_timeseries=RESULTS + "plots/generation_timeseries_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        total_generation=RESULTS + "plots/total_generation_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
    log:
        LOGS + "make_gb_plots/_s{simpl}_{clusters}_ec_l{ll}_{opts}.log",
    benchmark:
        BENCHMARKS + "make_gb_plots/_s{simpl}_{clusters}_ec_l{ll}_{opts}",
    threads: 1
    resources:
        mem_mb=4000,
    script:
        "../scripts/make_gb_plots.py"