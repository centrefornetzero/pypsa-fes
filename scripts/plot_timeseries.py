
import logging

logger = logging.getLogger(__name__)

import pandas as pd
import matplotlib.pyplot as plt
import pypsa

from _helpers import override_component_attrs
from make_summary import assign_carriers, assign_locations 



if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("plot_timeseries")

    tech_colors = snakemake.config["plotting"]["tech_colors"]
    tech_colors["wind"] = tech_colors["onwind"]
    tech_colors["H2 dispatch"] = tech_colors["H2 Fuel Cell"]
    tech_colors["H2 charge"] = tech_colors["H2 Electrolysis"]
    tech_colors["battery dispatch"] = tech_colors["battery"]
    tech_colors["battery charge"] = tech_colors["BEV charger"]
    tech_colors["DC"] = tech_colors["HVDC links"]

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    assign_carriers(n)
    assign_locations(n)

    freq = snakemake.config["plotting"]["timeseries_freq"]
    month = snakemake.config["plotting"]["timeseries_month"]
    
    target_files = ["gb", "scotland", "england"]
    bus_names = ["all", "GB0 Z5", "GB0 Z11"]

    template = "timeseries_{}_{}"

    for buses, target in zip(bus_names, target_files):

        if buses == "all":
            buses = pd.Index(n.buses.location.unique())
            buses = buses[buses.str.contains("GB")]
        
        else:
            buses = pd.Index([buses])

        load = n.loads_t.p_set.loc[:, buses].sum(axis=1)
        inflow = pd.DataFrame(index=n.snapshots)
        outflow = pd.DataFrame(index=n.snapshots)
                        
        print(n.links_t.p0.sum())
        n.links_t.p0.to_csv("p0.csv")
        print(n.links_t.p1.sum())
        n.links_t.p1.to_csv("p0.csv")
        n.generators.to_csv("genz_after.csv")

        for c in n.iterate_components(n.one_port_components | n.branch_components):

            print(c.name)

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
            
            elif c.name == "Link":
                
                idx_in = c.df.loc[c.df.bus1.isin(buses)].index
                c_energies = c.pnl.p0.loc[:, idx_in].multiply(c.df.loc[idx_in, "efficiency"], axis=1)
                
                for carrier in c.df.loc[idx_in, "carrier"].unique():

                    cols = c.df.loc[idx_in].loc[c.df.loc[idx_in, "carrier"] == carrier].index
                    if carrier in ["OCGT", "CCGT"]:                   
                        print(carrier)
                        print(c_energies.loc[:, cols].sum().rename(carrier))
                    
                    inflow[carrier] = (
                        c_energies
                        .loc[:, cols]
                        .sum(axis=1)
                    )
                
                idx_out = c.df.loc[c.df.bus0.isin(buses)].index
                c_energies = c.pnl.p1.loc[:, idx_out].multiply(c.df.loc[idx_out, "efficiency"], axis=1)

                for carrier in c.df.loc[idx_out, "carrier"].unique():
                    cols = c.df.loc[idx_out].loc[c.df.loc[idx_out, "carrier"] == carrier].index
                    
                    outflow[carrier] = c_energies.loc[:, cols].sum(axis=1)
                            
            
        print(inflow.head())         
            
        
        total = (
            (inflow.drop(columns="load").sum(axis=1) + outflow.sum(axis=1) - load)
            .resample(freq).mean()
        )
            
        
        from plot_gb_validation import stackplot_to_ax

        fig, ax = plt.subplots(1, 1, figsize=(16, 4))

        stackplot_to_ax(
            inflow.resample(freq).mean().drop(columns="load"),
            ax=ax,
            color_mapper=tech_colors,
            )
        
        if not outflow.empty:
            stackplot_to_ax(
                outflow.resample(freq).mean(),
                ax=ax,
                color_mapper=tech_colors,
                )
        
        ax.plot(total.index,
                total,
                color="black",
                label="Kirchhoff Check",
                linestyle="--",
                )        

        
        # ax.set_title(f"{c.name} {target} {buses}")
        ax.legend()

        print("saving at")
        print(snakemake.output[0])
        # plt.savefig(snakemake.output[0])
        plt.savefig('results/testfifth.pdf')

        plt.show()


        break
                    
                
                




