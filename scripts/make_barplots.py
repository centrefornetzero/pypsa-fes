import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("summarize_gb")

    logging.basicConfig(level=snakemake.config["logging"]["level"])

    networks_dict = {
        (gb_regions, ll, opts, flexopts, fes, year): "results/"
        + snakemake.params.RDIR
        + f"timeseries/timeseries-inflow_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.csv"
        for simpl in snakemake.config["scenario"]["simpl"]
        for gb_regions in snakemake.config["scenario"]["gb_regions"]
        for ll in snakemake.config["scenario"]["ll"]
        for opts in snakemake.config["scenario"]["opts"]
        for flexopts in snakemake.config["scenario"]["flexopts"]
        for fes in snakemake.config["scenario"]["fes"]
        for year in snakemake.config["scenario"]["year"]
    }

    included_carriers = ["winter flex", "regular flex", "thermal inertia", "V2G"]

    time_choice = "all"

    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["gb_regions", "ll", "opt", "flexopts", "fes", "year"]
    )
    df = pd.DataFrame(columns=columns, index=included_carriers)

    for col, filename in networks_dict.items():
        ts = pd.read_csv(filename, index_col=0, squeeze=True, parse_dates=True)

        if not time_choice == "all":
            raise NotImplementedError("time_choice != 'all' not implemented")
    
        print(ts.columns)
        df[col] = ts[included_carriers].sum()

    # df = df.mul(1 / df.sum(), axis=1)
    # print(df)

    df.columns = df.columns.get_level_values("year")
    print(df)

    df.to_csv(snakemake.output[0])