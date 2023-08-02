
import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import pypsa

from _helpers import override_component_attrs
from make_summary import assign_carriers, assign_locations

idx = pd.IndexSlice

opt_name = {"Store": "e", "Line": "s", "Transformer": "s"}


# def calculate_total_generation(n, label, df, snapshots):

def calculate_total_generation(n):
    """Calculate total generation for each carrier in GB."""

    buses = pd.Index(n.buses.location.unique())
    buses = buses[buses.str.contains("GB")]

    inflow = pd.DataFrame(index=n.snapshots)
    outflow = pd.DataFrame(index=n.snapshots)

    for c in n.iterate_components(n.one_port_components | n.branch_components):

        if c.name == "Load":
            continue

        if c.name in ["Generator", "StorageUnit"]: 
            idx = c.df.loc[c.df.bus.isin(buses)].index

            c_energies = (
                c.pnl.p
                .loc[:, idx]
                .multiply(n.snapshot_weightings.generators, axis=0)
            )

            for carrier in c.df.loc[idx, "carrier"].unique():
                cols = c.df.loc[idx].loc[c.df.loc[idx, "carrier"] == carrier].index

                inflow[carrier] = c_energies.loc[:, cols].sum(axis=1)

        elif c.name in ["Link", "Line"]:

            inout = ((c.df.bus0.isin(buses)) ^ (c.df.bus1.isin(buses))).rename("choice")

            subset = c.df.loc[inout]
            asbus0 = subset.loc[subset.bus0.isin(buses)].index
            asbus1 = subset.loc[subset.bus1.isin(buses)].index

            inout = pd.concat((
                - c.pnl.p0[asbus0],
                - c.pnl.p1[asbus1]
            ), axis=1)

            for carrier in c.df.loc[subset.index, "carrier"].unique():
                flow = inout[subset.loc[subset.carrier == carrier].index].sum(axis=1)

                inflow[carrier] = np.maximum(flow.values, 0.)
                outflow[carrier] = np.minimum(flow.values, 0.)

            if c.name == "Link":
                logger.warning("Hardcoded data gathering of DAC")
                dac = c.df.loc[c.df.carrier == "DAC"]
                subset = dac.loc[dac.bus2.isin(buses)]

                outflow["DAC"] = - c.pnl.p2[subset.index].sum(axis=1)


    inflow *= 1e-3
    outflow *= 1e-3

    return inflow, outflow



def to_csv(df):
    for key in df:
        df[key].to_csv(snakemake.output[key])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("make_summary")

    logging.basicConfig(level=snakemake.config["logging"]["level"])

    networks_dict = {
        (gb_regions, ll, opts, flexopts, fes, year): "results/"
        + snakemake.params.RDIR
        + f"networks/elec_s{simpl}_{gb_regions}_ec_l{ll}_{opts}_{flexopts}_{fes}_{year}.nc"
        for simpl in snakemake.config["scenario"]["simpl"]
        for gb_regions in snakemake.config["scenario"]["gb_regions"]
        for ll in snakemake.config["scenario"]["ll"]
        for opts in snakemake.config["scenario"]["opts"]
        for flexopts in snakemake.config["scenario"]["flexopts"]
        for fes in snakemake.config["scenario"]["fes"]
        for year in snakemake.config["scenario"]["year"]
    }


    from itertools import product


    Nyears = len(pd.date_range(freq="h", **snakemake.config["snapshots"])) / 8760

    outputs = [
        "total_generation",
    ]

    def sort_mindex(index, networks_dict):
        
        df = pd.DataFrame({name: index.get_level_values(name) for name in index.names})
        df["network_names"] = list(networks_dict.values())
        df["sort_main"] = df["flexopts"].str.len() 

        df = df.sort_values(by=["sort_main", "year"], ascending=[False, True])

        new_dict = {
            tuple(row[index.names]): row["network_names"] for _, row in df.iterrows()
        }
        new_index = pd.MultiIndex.from_frame(df[index.names])

        return new_index, new_dict

    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["gb_regions", "ll", "opt", "flexopts", "fes", "year"]
    )

    columns, networks_dict = sort_mindex(columns, networks_dict)

    snapshots = None
    if snakemake.config["flexibility"]["select_generation_snapshots_by_flex"]:
        snapshots = {(scenario, year): None 
                    for scenario, year in product(
                        snakemake.config["scenario"]["fes"],
                        snakemake.config["scenario"]["year"],
                    )}

    year_index = list(columns.names).index("year")
    fes_index = list(columns.names).index("fes") 
    
    df = {}

    for output in outputs:
        df[output] = pd.DataFrame(columns=columns, dtype=float)

    for label, filename in networks_dict.items():
        logger.info(f"Make summary for scenario {label}, using {filename}")

        year = label[year_index]
        fes = label[fes_index]

        overrides = override_component_attrs(snakemake.input.overrides)
        n = pypsa.Network(filename, override_component_attrs=overrides)

        assign_carriers(n)
        assign_locations(n)

        for output in outputs:
            inflow, outflow = globals()["calculate_" + output](n)

            if snapshots[(fes, year)] is None:
                
                mask = inflow[["regular flex", "winter flex"]].sum(axis=1) > 1.5
                snapshots[(fes, year)] = inflow[mask].index 

                import matplotlib.pyplot as plt

                if year == 2040:
                    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

                    ax.plot(snapshots[(fes, year)],
                            inflow.loc[snapshots[(fes, year)], ["regular flex", "winter flex"]].sum(axis=1), label="inflow")
                    plt.savefig("flexoverview.pdf")
                    plt.show()

                print("====================================================")
                print(label)
                print("Setting snapshots")
                print(snapshots[(fes, year)])
                print("We took {} percent of snapshots".format(len(snapshots[(fes, year)])/8760))
            
            
                
            inflow = inflow.loc[snapshots[(fes, year)]]
            outflow = outflow.loc[snapshots[(fes, year)]]

            df[output] = df[output].reindex(
                df[output].index.union(
                    list(set(inflow.columns.tolist() + outflow.columns.tolist())) 
                ),
                fill_value=0.
            )

            df[output].loc[inflow.columns, label] += (
                np.maximum(inflow.values, np.zeros_like(inflow.values))
                .sum(axis=0)
                )

    # df.loc[outflow.columns, label] += outflow.abs().sum()
    # df.loc[inflow.columns, label] += inflow.abs().sum()

    # return df

    to_csv(df)
