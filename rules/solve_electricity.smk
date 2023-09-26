# SPDX-FileCopyrightText: : 2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


rule solve_network:
    input:
        network=RESOURCES + "networks/elec_s{simpl}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.nc",
        capacity_constraints=RESOURCES + "fes_capacity_constraints_{fes}_{year}.csv",
        overrides="data/override_component_attrs",
    output:
        network=RESULTS + "networks/elec_s{simpl}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.nc",
    log:
        solver=normpath(
            LOGS + "solve_network/elec_s{simpl}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}_solver.log"
        ),
        python=LOGS
        + "solve_network/elec_s{simpl}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}_python.log",
    benchmark:
        BENCHMARKS + "solve_network/elec_s{simpl}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}"
    threads: 1
    resources:
        mem_mb=memory,
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"
