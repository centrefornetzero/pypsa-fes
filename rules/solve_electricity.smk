# SPDX-FileCopyrightText: : 2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


rule solve_network:
    input:
        network=RESOURCES + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.nc",
        capacity_constraints=RESOURCES + "fes_capacity_constraints_{fes}_{year}.csv",
        overrides="data/override_component_attrs",
    output:
        network=RESULTS + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.nc",
    log:
        solver=normpath(
            LOGS + "solve_network/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}_solver.log"
        ),
        python=LOGS
        + "solve_network/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}_python.log",
    benchmark:
        BENCHMARKS + "solve_network/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}"
    threads: 4
    resources:
        mem_mb=memory,
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"


rule solve_operations_network:
    input:
        network=RESULTS + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{fes}_{year}.nc",
    output:
        # network=RESULTS + "networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_op.nc",
        network=RESULTS + "networks/op_elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{fes}_{year}.nc",
    log:
        solver=normpath(
            LOGS
            + "solve_operations_network/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{fes}_{year}_op_solver.log"
        ),
        python=LOGS
        + "solve_operations_network/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{fes}_{year}_op_python.log",
    benchmark:
        (
            BENCHMARKS
            + "solve_operations_network/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{fes}_{year}"
        )
    threads: 4
    resources:
        mem_mb=(lambda w: 5000 + 372 * 50),
        # mem_mb=memory,
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_operations_network.py"
