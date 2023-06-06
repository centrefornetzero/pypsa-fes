# SPDX-FileCopyrightText: : 2023 Lukas Franken
#
# SPDX-License-Identifier: MIT

rule make_gb_summary:
    input:
        regions_onshore=RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
        network=RESULTS + "networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
    output:
        regions_map=RESULTS + "plots/network_regions_s{simpl}_{clusters}_ec_l{ll}_{opts}.png"
        capacity_expansion=RESULTS + "plots/capacity_expansion_s{simpl}_{clusters}_ec_l{ll}_{opts}.png"
        generation_timeseries=RESULTS + "plots/generation_timeseries_s{simpl}_{clusters}_ec_l{ll}_{opts}.png"
        total_generation=RESULTS + "plots/total_generation_s{simpl}_{clusters}_ec_l{ll}_{opts}.png"
    log:
        LOGS + "make_gb_plots/_s{simpl}_{clusters}_ec_l{ll}_{opts}.log"
    benchmark:
        BENCHMARKS + "make_gb_plots/_s{simpl}_{clusters}_ec_l{ll}_{opts}"
    threads: 1
    resources:
        mem_mb=4000,
    script:
        "../scripts/make_gb_plots.py"