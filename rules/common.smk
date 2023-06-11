# SPDX-FileCopyrightText: : 2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


def memory(w):
    factor = 3.0
    print(w)
    for o in w.opts.split("-"):
        m = re.match(r"^(\d+)h$", o, re.IGNORECASE)
        if m is not None:
            factor /= int(m.group(1))
            break
    for o in w.opts.split("-"):
        m = re.match(r"^(\d+)seg$", o, re.IGNORECASE)
        if m is not None:
            factor *= int(m.group(1)) / 8760
            break
    # if w.clusters.endswith("m"):
    #     return int(factor * (18000 + 180 * int(w.clusters[:-1])))
    # elif w.clusters == "all":
    #     return int(factor * (18000 + 180 * 4000))
    # else:
    #     return int(factor * (10000 + 195 * int(w.clusters)))
    if w.gb_regions.endswith("m"):
        return int(factor * (18000 + 180 * int(w.gb_regions[:-1])))
    elif w.gb_regions == "all":
        return int(factor * (18000 + 180 * 4000))
    elif w.gb_regions == "eso":
        return int(factor * (10000 + 195 * 31))
    elif w.gb_regions == "dno":
        return int(factor * (10000 + 195 * 25))
    else:
        return int(factor * (10000 + 195 * 50))



def input_eurostat(w):
    # 2016 includes BA, 2017 does not
    report_year = config["energy"]["eurostat_report_year"]
    return f"data/eurostat-energy_balances-june_{report_year}_edition"


def solved_previous_horizon(wildcards):
    planning_horizons = config["scenario"]["planning_horizons"]
    i = planning_horizons.index(int(wildcards.planning_horizons))
    planning_horizon_p = str(planning_horizons[i - 1])
    return (
        RESULTS
        + "postnetworks/elec_s{simpl}_{gb_regions}_l{ll}_{opts}_{sector_opts}_"
        + planning_horizon_p
        + ".nc"
    )
