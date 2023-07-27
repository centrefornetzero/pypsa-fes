# SPDX-FileCopyrightText: : 2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


localrules:
    copy_config,
    copy_conda_env,


rule plot_network:
    input:
        overrides="data/override_component_attrs",
        network=RESULTS + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.nc",
        regions=RESOURCES + "regions_onshore_elec_s{simpl}_{gb_regions}.geojson",
    output:
        map=RESULTS
        + "maps/elec_s{simpl}_{gb_regions}_l{ll}_{opts}_{flexopts}-costs-all_{fes}_{year}.pdf",
        today=RESULTS
        + "maps/elec_s{simpl}_{gb_regions}_l{ll}_{opts}_{flexopts}_{fes}_{year}-today.pdf",
    threads: 2
    resources:
        mem_mb=10000,
    benchmark:
        (
            BENCHMARKS
            + "plot_network/elec_s{simpl}_{gb_regions}_l{ll}_{opts}_{flexopts}_{fes}_{year}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_network.py"


rule copy_config:
    params:
        RDIR=RDIR,
    output:
        RESULTS + "config/config.yaml",
    threads: 1
    resources:
        mem_mb=1000,
    benchmark:
        BENCHMARKS + "copy_config"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/copy_config.py"


rule copy_conda_env:
    output:
        RESULTS + "config/environment.yaml",
    threads: 1
    resources:
        mem_mb=500,
    log:
        LOGS + "copy_conda_env.log",
    benchmark:
        BENCHMARKS + "copy_conda_env"
    conda:
        "../envs/environment.yaml"
    shell:
        "conda env export -f {output} --no-builds"


rule make_summary:
    params:
        RDIR=RDIR,
    input:
        overrides="data/override_component_attrs",
        networks=expand(
            RESULTS
            + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.nc",
            **config["scenario"]
        ),
        costs="data/costs_{}.csv".format(config["costs"]["year"])
        if config["foresight"] == "overnight"
        else "data/costs_{}.csv".format(config["scenario"]["planning_horizons"][0]),
        # plots=expand(
        #     RESULTS
        #     + "maps/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_-costs-all_{fes_scenario}_{planning_horizons}.pdf",
        #     **config["scenario"]
        # ),
    output:
        nodal_costs=RESULTS + "csvs/nodal_costs.csv",
        nodal_capacities=RESULTS + "csvs/nodal_capacities.csv",
        nodal_cfs=RESULTS + "csvs/nodal_cfs.csv",
        cfs=RESULTS + "csvs/cfs.csv",
        costs=RESULTS + "csvs/costs.csv",
        capacities=RESULTS + "csvs/capacities.csv",
        curtailment=RESULTS + "csvs/curtailment.csv",
        energy=RESULTS + "csvs/energy.csv",
        supply=RESULTS + "csvs/supply.csv",
        supply_energy=RESULTS + "csvs/supply_energy.csv",
        prices=RESULTS + "csvs/prices.csv",
        weighted_prices=RESULTS + "csvs/weighted_prices.csv",
        market_values=RESULTS + "csvs/market_values.csv",
        price_statistics=RESULTS + "csvs/price_statistics.csv",
        metrics=RESULTS + "csvs/metrics.csv",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        LOGS + "make_summary.log",
    benchmark:
        BENCHMARKS + "make_summary"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/make_summary.py"


rule plot_summary:
    params:
        RDIR=RDIR,
    input:
        costs=RESULTS + "csvs/costs.csv",
        nodal_costs=RESULTS + "csvs/nodal_costs.csv",
        energy=RESULTS + "csvs/energy.csv",
        balances=RESULTS + "csvs/supply_energy.csv",
        nodal_capacities=RESULTS + "csvs/nodal_capacities.csv",
        # eurostat=input_eurostat,
    output:
        costs=RESULTS + "graphs/costs.pdf",
        energy=RESULTS + "graphs/energy.pdf",
        balances=RESULTS + "graphs/balances-energy.pdf",
        capacities=RESULTS + "graphs/capacities.pdf",    
    threads: 2
    resources:
        mem_mb=10000,
    log:
        LOGS + "plot_summary.log",
    benchmark:
        BENCHMARKS + "plot_summary"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_summary.py"


rule plot_timeseries:
    params:
        RDIR=RDIR,
    input:
        network=RESULTS + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.nc",
        overrides="data/override_component_attrs",
    output:
        timeseries_gb_year=RESULTS + "graphs/timeseries-year-gb_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.pdf",
        timeseries_gb_short=RESULTS + "graphs/timeseries-short-gb_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.pdf",
        timeseries_scotland_year=RESULTS + "graphs/timeseries-year-scotland_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.pdf",
        timeseries_scotland_short=RESULTS + "graphs/timeseries-short-scotland_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.pdf",
        timeseries_england_year=RESULTS + "graphs/timeseries-year-england_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.pdf",
        timeseries_england_short=RESULTS + "graphs/timeseries-short-england_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.pdf",
        # emission_timeseries=RESULTS + "graphs/timeseries-emissions_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{fes_scenario}_{planning_horizons}.pdf",
        co2_barplot=RESULTS + "graphs/barplot-co2_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.pdf",
    threads: 1
    resources:
        mem_mb=10000,
    log:
        LOGS + "plot_timeseries_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.log",
    benchmark:
        BENCHMARKS + "plot_timeseries_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_timeseries.py"