# SPDX-FileCopyrightText: : 2023 Lukas Franken
#
# SPDX-License-Identifier: MIT


rule build_flex_network:
    input:
        network=RESOURCES + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.nc",
        flex_data="data/turndown_events/turndown_{td_event}.csv",
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


rule plot_gb_validation:
    params:
        RDIR=RDIR
    input:
        gb_generation_data="data/gb_generation_2022.csv",
        network=RESULTS + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.nc",
    output:
        generation_timeseries_year=RESULTS + "plots/generation_timeseries_year_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.pdf",
        generation_timeseries_weeks=RESULTS + "plots/generation_timeseries_weeks_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.pdf",
        total_generation=RESULTS + "plots/total_generation_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.pdf",
        # regions_map=RESULTS + "plots/network_regions_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        # capacity_expansion=RESULTS + "plots/capacity_expansion_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        # generation_timeseries=RESULTS + "plots/generation_timeseries_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        # total_generation=RESULTS + "plots/total_generation_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
    log:
        LOGS + "plot_gb_validation/_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.log",
    benchmark:
        BENCHMARKS + "plot_gb_validation/_s{simpl}_{gb_regions}_ec_l{ll}_{opts}",
    threads: 1
    resources:
        mem_mb=4000,
    script:
        "../scripts/plot_gb_validation.py"


rule plot_saving_sessions:
    input:
        network=RESULTS + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.nc",
        saving_sessions_data="data/ss_turndown.csv",
    output:
        generation_timeseries=RESULTS + "plots/timeseries_generation_saving_timings_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.pdf",
        saved_fuel=RESULTS + "plots/model_fuel_savings_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.pdf",
        saved_emissions=RESULTS + "plots/model_emission_savings_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.pdf",
    log:
        LOGS + "plot_saving_sessions/_s{simpl}_{gb_regions}_ec_l{ll}_{opts}.log",
    benchmark:
        BENCHMARKS + "plot_saving_sessions/_s{simpl}_{gb_regions}_ec_l{ll}_{opts}",
    threads: 1
    resources:
        mem_mb=4000,
    script:
        "../scripts/plot_saving_sessions.py"
    