# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors and Lukas Franken
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such
as.

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** per tonne emissions of carbon-dioxide (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours
  or segmenting time series into chunks of varying lengths using ``tsam``.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        version:
        fill_values:
        emission_prices:
        marginal_cost:
        capital_cost:

    electricity:
        co2limit:
        max_hours:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`costs_cf`, :ref:`electricity_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``networks/elec_s{simpl}_{clusters}.nc``: confer :ref:`cluster`

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Complete PyPSA network that will be handed to the ``solve_network`` rule.

Description
-----------

.. tip::
    The rule :mod:`prepare_elec_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`prepare_network`.
"""

import logging
import re

import pypsa
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product

from _helpers import (
    configure_logging,
    generate_periodic_profiles,
    override_component_attrs,
)
from add_electricity import load_costs, update_transmission_costs, calculate_annuity
from prepare_sector_network import cycling_shift, prepare_costs
from _fes_helpers import (
    get_data_point,
    scenario_mapper,
    get_gb_total_number_cars,
    get_gb_total_transport_demand,
    get_smart_charge_v2g,
    get_power_generation_emission,
    get_battery_capacity,
    ) 

from pypsa.descriptors import expand_series

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def add_co2limit(n, co2limit, Nyears=1.0):
    n.add(
        "GlobalConstraint",
        "CO2Limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2limit * Nyears,
    )


def add_gaslimit(n, gaslimit, Nyears=1.0):
    sel = n.carriers.index.intersection(["OCGT", "CCGT", "CHP"])
    n.carriers.loc[sel, "gas_usage"] = 1.0

    n.add(
        "GlobalConstraint",
        "GasLimit",
        carrier_attribute="gas_usage",
        sense="<=",
        constant=gaslimit * Nyears,
    )


def add_emission_prices(n, emission_prices={"co2": 0.0}, exclude_co2=False):
    if exclude_co2:
        emission_prices.pop("co2")
    ep = (
        pd.Series(emission_prices).rename(lambda x: x + "_emissions")
        * n.carriers.filter(like="_emissions")
    ).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep


def add_emission_prices_t(n):
    co2_price = pd.read_csv(snakemake.input.co2_price, index_col=0,
                            parse_dates=True)

    co2_price = co2_price[~co2_price.index.duplicated()]
    co2_price = co2_price.reindex(n.snapshots).fillna(method="ffill").fillna(method="bfill")
    emissions = n.generators.carrier.map(n.carriers.co2_emissions)
    co2_cost = (expand_series(emissions, n.snapshots).T
                .mul(co2_price.iloc[:,0], axis=0))
    n.generators_t.marginal_cost += (co2_cost.reindex(columns=n.generators_t.marginal_cost.columns))


def set_line_s_max_pu(n, s_max_pu=0.7):
    n.lines["s_max_pu"] = s_max_pu
    logger.info(f"N-1 security margin of lines set to {s_max_pu}")


def set_transmission_limit(n, ll_type, factor, costs, Nyears=1):
    links_dc_b = n.links.carrier == "DC" if not n.links.empty else pd.Series()

    _lines_s_nom = (
        np.sqrt(3)
        * n.lines.type.map(n.line_types.i_nom)
        * n.lines.num_parallel
        * n.lines.bus0.map(n.buses.v_nom)
    )
    lines_s_nom = n.lines.s_nom.where(n.lines.type == "", _lines_s_nom)

    col = "capital_cost" if ll_type == "c" else "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + n.links.loc[links_dc_b, "p_nom"] @ n.links.loc[links_dc_b, col]
    )

    update_transmission_costs(n, costs)

    if factor == "opt" or float(factor) > 1.0:
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True

        n.links.loc[links_dc_b, "p_nom_min"] = n.links.loc[links_dc_b, "p_nom"]
        n.links.loc[links_dc_b, "p_nom_extendable"] = True

    if factor != "opt":
        con_type = "expansion_cost" if ll_type == "c" else "volume_expansion"
        rhs = float(factor) * ref
        n.add(
            "GlobalConstraint",
            f"l{ll_type}_limit",
            type=f"transmission_{con_type}_limit",
            sense="<=",
            constant=rhs,
            carrier_attribute="AC, DC",
        )

    return n


def average_every_nhours(n, offset):
    logger.info(f"Resampling the network to {offset}")
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                pnl[k] = df.resample(offset).mean()

    return m


def apply_time_segmentation(n, segments, solver_name="cbc"):
    logger.info(f"Aggregating time series to {segments} segments.")
    try:
        import tsam.timeseriesaggregation as tsam
    except:
        raise ModuleNotFoundError(
            "Optional dependency 'tsam' not found." "Install via 'pip install tsam'"
        )

    p_max_pu_norm = n.generators_t.p_max_pu.max()
    p_max_pu = n.generators_t.p_max_pu / p_max_pu_norm

    load_norm = n.loads_t.p_set.max()
    load = n.loads_t.p_set / load_norm

    inflow_norm = n.storage_units_t.inflow.max()
    inflow = n.storage_units_t.inflow / inflow_norm

    raw = pd.concat([p_max_pu, load, inflow], axis=1, sort=False)

    agg = tsam.TimeSeriesAggregation(
        raw,
        hoursPerPeriod=len(raw),
        noTypicalPeriods=1,
        noSegments=int(segments),
        segmentation=True,
        solver=solver_name,
    )

    segmented = agg.createTypicalPeriods()

    weightings = segmented.index.get_level_values("Segment Duration")
    offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
    snapshots = [n.snapshots[0] + pd.Timedelta(f"{offset}h") for offset in offsets]

    n.set_snapshots(pd.DatetimeIndex(snapshots, name="name"))
    n.snapshot_weightings = pd.Series(
        weightings, index=snapshots, name="weightings", dtype="float64"
    )

    segmented.index = snapshots
    n.generators_t.p_max_pu = segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm
    n.loads_t.p_set = segmented[n.loads_t.p_set.columns] * load_norm
    n.storage_units_t.inflow = segmented[n.storage_units_t.inflow.columns] * inflow_norm

    return n


def enforce_autarky(n, only_crossborder=False):
    if only_crossborder:
        lines_rm = n.lines.loc[
            n.lines.bus0.map(n.buses.country) != n.lines.bus1.map(n.buses.country)
        ].index
        links_rm = n.links.loc[
            n.links.bus0.map(n.buses.country) != n.links.bus1.map(n.buses.country)
        ].index
    else:
        lines_rm = n.lines.index
        links_rm = n.links.loc[n.links.carrier == "DC"].index
    n.mremove("Line", lines_rm)
    n.mremove("Link", links_rm)


def set_line_nom_max(n, s_nom_max_set=np.inf, p_nom_max_set=np.inf):
    n.lines.s_nom_max.clip(upper=s_nom_max_set, inplace=True)
    n.links.p_nom_max.clip(upper=p_nom_max_set, inplace=True)


# TODO: PyPSA-Eur merge issue
def remove_elec_base_techs(n):
    """
    Remove conventional generators (e.g. OCGT) and storage units (e.g.
    batteries and H2) from base electricity-only network, since they're added
    here differently using links.
    """
    logger.warning("Removing conventional generators and storage units in GB only!")
    for c in n.iterate_components(snakemake.config["pypsa_eur"]):
        to_keep = snakemake.config["pypsa_eur"][c.name]
        to_remove = pd.Index(c.df.carrier.unique()).symmetric_difference(to_keep)
        if to_remove.empty:
            continue
        logger.info(f"Removing {c.list_name} with carrier {list(to_remove)}")
        names = c.df.index[c.df.carrier.isin(to_remove)]

        subset = c.df.loc[names]
        names = subset.index[subset.index.str.contains("GB")]

        n.mremove(c.name, names)
        # n.carriers.drop(to_remove, inplace=True, errors="ignore")


def scale_generation_capacity(n, capacity_file):
    
    generation_mapper = {
        "gas": ["OCGT", "CCGT"],
        "coal": ["coal", "lignite"],
        "nuclear": ["nuclear"],
        "biomass": ["biomass"],
    }

    constraints = pd.read_csv(capacity_file, index_col=1)["value"]
    gb_gen = n.generators.loc[n.generators.bus.str.contains("GB")]

    print("==========================")
    print(constraints)
    print("==========================")

    for fes_gen, target in generation_mapper.items():
        
        index = gb_gen[gb_gen.carrier.isin(target)].index

        print(target)
        # if not fes_gen in constraints.index or constraints.loc[fes_gen] == 0.:
        if fes_gen not in constraints.index:
            print("ignoring that")
            continue

        if constraints.loc[fes_gen] == 0.:
            
            logger.info(f"dropping generators \n {index}")
            
            if index[0] in n.generators_t.marginal_cost.columns:
                n.generators_t.marginal_cost.drop(index, axis=1, inplace=True)

            n.generators.drop(index, inplace=True)
            continue

        print(target, constraints.loc[fes_gen])

        adapted = n.generators.loc[index, "p_nom"].copy()
        adapted *= constraints.loc[fes_gen] / adapted.sum() 

        n.generators.loc[index, "p_nom"] = adapted
        n.generators.loc[index, "p_nom_min"] = adapted


def convert_generators_to_links(n, costs):

    # logger.info("Converting generators to links")

    gb_generation = n.generators.loc[n.generators.bus.str.contains("GB")]
    conventionals = {
        "CCGT": "gas",
        "OCGT": "gas",
        "coal": "coal",
        "lignite": "lignite",
        "biomass": "biomass",
    }

    buses_to_add = list()

    for generator, carrier in conventionals.items():

        gens = gb_generation.loc[gb_generation.carrier == generator]

        if not f"GB_{carrier}_bus" in n.buses.index:
            buses_to_add.append(carrier)

        if gens.empty:
            print(f"nothing to do for carrier {generator}")
            continue

        n.madd(
            "Link",
            gens.index,
            bus0=f"GB_{carrier}_bus",
            bus1=gens.bus,
            bus2="gb co2 atmosphere",
            marginal_cost=costs.at[generator, "efficiency"]
            * costs.at[generator, "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at[generator, "efficiency"]
            * costs.at[generator, "fixed"],  # NB: fixed cost is per MWel
            carrier=generator,
            p_nom=gens.p_nom.values / costs.at[generator, "efficiency"],
            efficiency=costs.at[generator, "efficiency"],
            efficiency2=costs.at[carrier, "CO2 intensity"],
            # lifetime=costs.at[generator, "lifetime"],
        )

    logger.warning((
        "What is added is not affected by `remove_elec_base_techs` due to "
        "'coal', 'lignite', 'CCGT', 'OCGT' being in config[`pypsa_eur`]" 
        ))

    remove_elec_base_techs(n)

    for carrier in set(buses_to_add):
        
        if carrier == "biomass": continue

        print(f"Adding bus GB_{carrier}_bus, and generator GB_{carrier}")

        n.add("Bus",
            f"GB_{carrier}_bus",
            carrier=carrier)

        n.add("Generator",
                f"GB_{carrier}",
                bus=f"GB_{carrier}_bus",
                carrier=carrier,
                # p_nom_extandable=True,
                p_nom=1e6,
                marginal_cost=costs.at[carrier, "fuel"],
                )

def add_gas_ccs(n, costs):

    gb_buses = pd.Index(n.generators.loc[n.generators.bus.str.contains("GB")].bus.unique())
    gb_buses = n.buses.loc[gb_buses].loc[n.buses.loc[gb_buses].carrier == "AC"].index

    logger.warning("Adding Allam cycle...")

    n.madd(
        "Link",
        gb_buses,
        suffix=" allam",
        bus0=f"GB_gas_bus",
        bus1=gb_buses,
        bus2="gb co2 stored",
        # bus3="gb co2 atmosphere",
        carrier="allam",
        p_nom_extendable=True,
        # TODO: add costs to technology-data
        capital_cost=0.6 * 1.5e6 * 0.1,  # efficiency * EUR/MW * annuity
        marginal_cost=2,
        efficiency=0.6,
        efficiency2=costs.at["gas", "CO2 intensity"],
        # efficiency3=-costs.at["gas", "CO2 intensity"],
        lifetime=30.,
    )

    """
    marginal_cost=costs.at["gas CCS", "efficiency"]
    * costs.at["gas CCS", "VOM"],  # NB: VOM is per MWel
    capital_cost=costs.at["gas CCS", "efficiency"]
    * costs.at["gas CCS", "fixed"],  # NB: fixed cost is per MWel
    efficiency=costs.at["gas CCS", "efficiency"],
    """

# adapted from `add_heat` method in `scripts/prepare_sector_network.py`
def add_heat_pump_load(
    n,
    heat_demand_file,
    ashp_cop_file,
    energy_totals_file,
    intraday_profile_file,
    scenario,
    year,
    ):

    year = int(year)

    intraday_profiles = pd.read_csv(intraday_profile_file, index_col=0)

    daily_space_heat_demand = (
        xr.open_dataarray(heat_demand_file)
        .to_pandas()
        .reindex(index=n.snapshots, method="ffill")
    )

    gb_regions = [col for col in daily_space_heat_demand if "GB" in col]
    daily_space_heat_demand = daily_space_heat_demand[gb_regions]

    pop_weighted_energy_totals = pd.read_csv(energy_totals_file, index_col=0)

    sectors = ["residential"]
    uses = ["water", "space"]

    heat_demand = {}

    for sector, use in product(sectors, uses):
        weekday = list(intraday_profiles[f"{sector} {use} weekday"])
        weekend = list(intraday_profiles[f"{sector} {use} weekend"])
        weekly_profile = weekday * 5 + weekend * 2
        intraday_year_profile = generate_periodic_profiles(
            daily_space_heat_demand.index.tz_localize("UTC"),
            nodes=daily_space_heat_demand.columns,
            weekly_profile=weekly_profile,
        )

        if use == "space":
            heat_demand_shape = daily_space_heat_demand * intraday_year_profile
        else:
            heat_demand_shape = intraday_year_profile

        heat_demand[f"{sector} {use}"] = (
            heat_demand_shape / heat_demand_shape.sum()
        ).multiply(pop_weighted_energy_totals[f"total {sector} {use}"]) * 1e6

    heat_demand = pd.concat(heat_demand, axis=1)
    heat_demand = pd.DataFrame({
        region: heat_demand[[col for col in heat_demand.columns if region in col]].sum(axis=1)
        for region in daily_space_heat_demand.columns
    })

    cop = xr.open_dataarray(ashp_cop_file).to_dataframe().iloc[:,0].unstack()
    cop = cop[gb_regions]

    cop = cop.rename(
        columns={old: [col for col in heat_demand.columns if col in old][0]
        for old in cop.columns}
    )

    # to electricity demand through cop
    heat_demand = heat_demand.divide(cop)

    # scale according to scenario
    # get number of elec load through residential heating
    hp_load_future = get_data_point("elec_demand_home_heating", scenario, year)
    hp_load_base = get_data_point("elec_demand_home_heating", scenario, 2020)

    heat_demand = (
        heat_demand 
        / heat_demand.sum().sum() 
        * (hp_load_future - hp_load_base) 
    )

    n.loads_t.p_set[gb_regions] += heat_demand


def add_bev(n, transport_config):
    """Adds BEV load and respective stores units;
    adapted from `add_land_transport` method in `scripts/prepare_sector_network.py`"""

    logger.info("Adding BEV load and respective storage units")

    year = int(snakemake.wildcards.year)

    bev_flexibility = "bev" in snakemake.wildcards.opts.split("-")
    
    transport = pd.read_csv(
        snakemake.input.transport_demand, index_col=0, parse_dates=True
    )
    number_cars = pd.read_csv(snakemake.input.transport_data, index_col=0)[
        "number cars"
    ]
    avail_profile = pd.read_csv(
        snakemake.input.avail_profile, index_col=0, parse_dates=True
    )
    dsm_profile = pd.read_csv(
        snakemake.input.dsm_profile, index_col=0, parse_dates=True
    )

    gb_cars_2050 = get_gb_total_number_cars(
        snakemake.input.fes_table, "FS") * 1e6

    gb_transport_demand = get_gb_total_transport_demand(
        snakemake.input.fes_table)

    bev_cars = get_data_point("bev_cars_on_road",
        snakemake.wildcards.fes,
        year)
    
    electric_share = bev_cars / gb_cars_2050
    logger.info(f"EV share: {np.around(electric_share*100, decimals=2)}%")

    gb_nodes = pd.Index([col for col in transport.columns if "GB" in col])

    if electric_share > 0.0:

        p_set = (
            electric_share
            * (
                transport[gb_nodes]
                + cycling_shift(transport[gb_nodes], 1)
                + cycling_shift(transport[gb_nodes], 2)
            )
            / 3
        )

        n.madd(
            "Bus",
            gb_nodes,
            location=gb_nodes,
            suffix=" EV battery",
            carrier="Li ion",
            unit="MWh_el",
        )

        n.madd(
            "Load",
            gb_nodes, 
            suffix = " land transport EV",
            bus=gb_nodes + " EV battery",
            carrier="land transport EV",
            p_set=p_set,
        )

        p_nom = (
            number_cars
            * transport_config.get("bev_charge_rate", 0.011)
            * electric_share
        )

        logger.info("Assuming BEV charge efficiency of 0.9")
        n.madd(
            "Link",
            gb_nodes,
            suffix=" BEV charger",
            bus0=gb_nodes,
            bus1=gb_nodes + " EV battery",
            p_nom=p_nom,
            carrier="BEV charger",
            p_max_pu=avail_profile[gb_nodes],
            efficiency=0.9,
            # These were set non-zero to find LU infeasibility when availability = 0.25
            # p_nom_extendable=True,
            # p_nom_min=p_nom,
            # capital_cost=1e6,  #i.e. so high it only gets built where necessary
        )

        smart_share, v2g_share = get_smart_charge_v2g(
            snakemake.input.fes_table,
            snakemake.wildcards.fes,
            year)
        
        if v2g_share > 0. and bev_flexibility:
            logger.info("Assuming V2G efficiency of 0.9")
            v2g_efficiency = 0.9
            n.madd(
                "Link",
                gb_nodes,
                suffix=" V2G",
                bus0=gb_nodes + " EV battery",
                bus1=gb_nodes,
                p_nom=p_nom * v2g_share,
                carrier="V2G",
                p_max_pu=avail_profile[gb_nodes],
                efficiency=v2g_efficiency,
            )

        if smart_share > 0. and bev_flexibility:
            logger.info("Assuming average capacity size of 0.05 MWh")
            avg_battery_size = 0.05
            e_nom = (
                number_cars
                * avg_battery_size
                * smart_share
                * electric_share
            )

            n.madd(
                "Store",
                gb_nodes,
                suffix=" battery storage",
                bus=gb_nodes + " EV battery",
                carrier="battery storage",
                e_cyclic=True,
                e_nom=e_nom,
                e_max_pu=1,
                e_min_pu=dsm_profile[gb_nodes],
            )


def add_gb_co2_tracking(n, net_change_co2):
    # can also be negative
    n.add(
        "Bus",
        "gb co2 atmosphere",
        carrier="co2",
        unit="t_co2",
    )

    e_max_pu = pd.Series(1., n.snapshots)
    e_max_pu.iloc[-1] = (-1.) ** (int(net_change_co2 < 0))

    print("Net change co2: ", net_change_co2)
    print("e_nom: ", abs(net_change_co2))

    n.add(
        "Store",
        "gb co2 atmosphere",
        carrier="co2",
        bus="gb co2 atmosphere",
        e_nom=abs(net_change_co2*1e6),
        # e_nom_extendable=True,
        e_min_pu=-1.,
        e_max_pu=e_max_pu,
        e_initial=0.,
    )


def add_dac(n, costs):

    # daccs_removal_target = abs(daccs_removal_target)

    gb_buses = n.buses.loc[n.buses.index.str.contains("GB")]
    gb_buses = gb_buses.loc[gb_buses.carrier == "AC"]

    logger.info("Adding direct air capture")
    logger.warning("Neglecting heat demand of direct air capture")

    # logger.warning("Changed sign of DAC efficiency to be positive")
    efficiency2 = - (
        costs.at["direct air capture", "electricity-input"]
        + costs.at["direct air capture", "compression-electricity-input"]
    )

    heat_demand = (
        costs.at["direct air capture", "heat-input"]
        - costs.at["direct air capture", "compression-heat-output"]
    )

    # ad-hoc estimation for the cost of low-grade heat if it was generated from 
    # natural gas, ccgt, heat pump
    mcost_heat = (
        costs.at["gas", "fuel"] / costs.at["CCGT", "efficiency"] +
        costs.at["CCGT", "VOM"]
    ) / costs.at["central air-sourced heat pump", "efficiency"] + costs.at["central air-sourced heat pump", "VOM"]

    logger.info(f"Estimating cost of heat for DAC to be {np.around(mcost_heat, decimals=2)} EUR/MWh_th")
    logger.info(f"Assuming required LHV heat net input {np.around(heat_demand, decimals=2)} MWh_th/tCO2")

    # this tracks GB CO2 stored, e.g. underground
    n.add(
        "Bus",
        "gb co2 stored", 
        carrier="co2 stored",
        unit="t_co2",
    )

    n.add(
        "Store",
        "gb co2 stored",
        e_nom_extendable=True,
        e_min_pu=-1.,
        carrier="co2",
        bus="gb co2 stored",
    )

    n.madd(
        "Link",
        gb_buses.index,
        suffix=" DAC",
        bus0="gb co2 atmosphere",
        bus1="gb co2 stored",
        bus2=gb_buses.index,
        carrier="DAC",
        capital_cost=costs.at["direct air capture", "fixed"],
        marginal_cost=heat_demand * mcost_heat,
        efficiency=1.0,
        efficiency2=efficiency2,
        p_nom_extendable=True,
        lifetime=costs.at["direct air capture", "lifetime"],
    )


def add_biogas(n, costs):

    gb_buses = n.buses.loc[n.buses.index.str.contains("GB")]
    gb_buses = gb_buses.loc[gb_buses.carrier == "AC"]

    biogas_potentials = pd.read_csv(snakemake.input.biomass_potentials, index_col=0)
    biogas_potentials = biogas_potentials.loc[gb_buses.index, "biogas"].sum()

    logger.info("Adding biogas with potentials")
    logger.info(f"Biogas potentials: {int(biogas_potentials*1e-6)} TWh_th")

    n.add("Bus",
        "GB_biomass_bus",
        carrier="biomass")
    
    n.add(
        "Store",
        "gb biomass",
        bus="GB_biomass_bus",
        carrier="biomass",
        e_nom=biogas_potentials,
        marginal_cost=costs.at["biogas", "fuel"],
        e_initial=biogas_potentials,
    )

    n.add(
        "Link",
        "biogas upgrading",
        bus0="GB_biomass_bus",
        bus1="GB_gas_bus",
        bus2="gb co2 atmosphere",
        carrier="biogas to gas",
        capital_cost=costs.loc["biogas upgrading", "fixed"],
        marginal_cost=costs.loc["biogas upgrading", "VOM"],
        efficiency=1.,
        efficiency2=-costs.at["gas", "CO2 intensity"],
        p_nom_extendable=True,
    )


def add_flexibility(n, mode):

    assert mode in ["winter", "regular"], f"chosen mode {mode} must be either 'winter' or 'regular'"
    name = mode + " flex"

    year = int(snakemake.wildcards.year)

    gb_buses = n.buses.loc[n.buses.index.str.contains("GB")]
    gb_buses = gb_buses.loc[gb_buses.carrier == "AC"]
    
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)     
    pop_layout = pop_layout.loc[pop_layout.index.str.contains("GB")]["total"]

    total_pop = pop_layout.sum()

    if year < snakemake.config["flexibility"]["completion_smartmeter_rollout"]:
        sm_rollout = np.interp(
            year,
            [2022, snakemake.config["flexibility"]["completion_smartmeter_rollout"]],
            [snakemake.config["flexibility"]["smartmeter_rollout_2022"]*1e-2, 1.],
            )
    else:
        sm_rollout = 1.

    opt_in_rate = snakemake.config["flexibility"][f"{mode}_opt_in_rate"] * 1e-2

    gb_population = 67_330_000
    avg_people_per_household = 2.36         # Statista 2022

    turndown_per_household = snakemake.config["flexibility"]["turndown_potential_per_household"] * 1e-3     # MWh
    turndown_per_person = turndown_per_household / avg_people_per_household

    weekly_allowance = 5 if mode == "winter" else 2

    start_hour = snakemake.config["flexibility"]["event_start_hour"]
    end_hour = snakemake.config["flexibility"]["event_end_hour"]

    single_event_capacity = (
        (pop_layout / total_pop * gb_population) 
        * turndown_per_person 
        * sm_rollout
        * opt_in_rate
    )

    # weekly recharge
    recharge_profile = pd.Series(0., index=n.snapshots)
    mask = (n.snapshots.hour == 0) & (n.snapshots.weekday == 1)
    recharge_profile.loc[mask] = 1.

    # events define time when demand flex can be used
    event_space = pd.DataFrame(0., index=n.snapshots, columns=gb_buses.index)

    mask = (n.snapshots.hour >= start_hour) & (n.snapshots.hour <= end_hour)

    if mode == "winter":
        mask = mask & (
            n.snapshots.month.isin(
                snakemake.config["flexibility"]["winter_months"])
            ) & (n.snapshots.weekday <= 5)

    event_space.loc[mask] = 1.

    n.add("Bus",
        name,
        carrier=name,
        )

    n.add("Generator",
        name,
        bus=name,
        carrier=name,
        p_nom=single_event_capacity.sum() * weekly_allowance,
        p_max_pu=recharge_profile,
        marginal_cost=0.,
    )

    n.add("Store",
        name,
        bus=name,
        carrier=name,
        e_nom=single_event_capacity.sum() * weekly_allowance,
    )

    n.madd("Link",
        gb_buses.index,
        suffix=f" {name}", 
        bus0=name,
        bus1=gb_buses.index,
        carrier=name,
        p_nom=single_event_capacity * weekly_allowance,
        p_max_pu=event_space,
        marginal_cost=0.,
    )

    # ensure no double flexibility in winter months
    if "winter flex" in n.buses.carrier and "regular flex" in n.buses.carrier:

        win = n.links.loc[n.links.carrier == "winter flex"].index
        reg = n.links.loc[n.links.carrier == "regular flex"].index

        if (n.links_t.p_max_pu[[win[0], reg[0]]].sum(axis=1) > 1.).any():

            # remove regular flex for winter months
            mask = n.links_t.p_max_pu[win[0]].astype(bool)
            n.links_t.p_max_pu.loc[mask, reg] = 0.


def add_batteries(n):
    """Adds battery storage (exluding V2G)

    Batteries are added as Storage Units
    Here, for PHS the distributions matches the distribution of 
    generation capacity or ror
    For batteries, LAES and CAES, the distribution is weighted by total load

    """

    gb_buses = pd.Index(n.generators.loc[n.generators.bus.str.contains("GB")].bus.unique())
    gb_buses = n.buses.loc[gb_buses].loc[n.buses.loc[gb_buses].carrier == "AC"].index

    year = int(snakemake.wildcards.year)
    scenario = snakemake.wildcards.fes

    p_noms, e_noms = get_battery_capacity(scenario, year)

    threshold = snakemake.config["flexibility"]["battery_ignore_threshold"]

    p_noms = p_noms.loc[p_noms > threshold]
    e_noms = e_noms.loc[p_noms.index]

    logger.info(f"Adding battery storage techs {list(p_noms.index)}") 

    tech = "PHS"
    logger.info(f"Adding battery storage tech {tech} with total capacity {p_noms.loc[tech]:.2f} GW")
        
    gb_ror = n.generators.loc[
        (n.generators.carrier == "ror") & 
        (n.generators.bus.isin(gb_buses))
        ]

    if not gb_ror.empty:

        buses = pd.Index(gb_ror.bus)

        weights = gb_ror.p_nom
        p_nom = weights / weights.sum() * p_noms.loc[tech]
        p_nom.index = buses + f" {tech}"

        efficiency_dispatch = n.storage_units.loc[n.storage_units.carrier == tech].efficiency_store.values[0]
        efficiency_store = efficiency_dispatch

        p_nom *= 1e3 # GW -> MW

        old_phs = n.storage_units.loc[
            (n.storage_units.bus.isin(gb_buses)) & 
            (n.storage_units.carrier == tech)
            ].index

        max_hours = n.storage_units.loc[old_phs].max_hours.mean()
        n.storage_units.drop(old_phs, inplace=True)

        n.madd(
            "StorageUnit",
            buses + " PHS",
            bus=buses,
            p_nom=p_nom,
            carrier=tech,
            efficiency_store=efficiency_store,
            efficiency_dispatch=efficiency_store,
            p_min_pu=-1.,
            max_hours=max_hours,
            cyclic_state_of_charge=True,
        )

    try:
        tech = p_noms.drop("PHS").index
    except KeyError:
        tech = p_noms.index

    if tech.empty:
        return

    buses = gb_buses
    
    weights = n.loads_t.p_set[gb_buses].sum()
    p_nom = weights / weights.sum() * p_noms.loc[tech].sum()
    e_nom = e_noms.loc[tech].sum() * 1e3 # GWh -> MWh
    
    p_nom *= 1e3 # GW -> MW

    efficiency = 0.97

    n.madd(
        "StorageUnit",
        buses,
        suffix=" grid battery",
        bus=buses,
        p_nom=p_nom,
        carrier="grid battery",
        efficiency_store=efficiency,
        efficiency_dispatch=efficiency,
        p_min_pu=-1.,
        max_hours=e_nom / p_nom.sum(),
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_network", simpl="", clusters="40", ll="v0.3", opts="Co2L-24H"
        )
    configure_logging(snakemake)

    opts = snakemake.wildcards.opts.split("-")

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input[0], override_component_attrs=overrides)

    fes = snakemake.wildcards.fes
    year = snakemake.wildcards.year
    logger.info(f"Preparing network for {fes} in {year}.")

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0

    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.config["costs"],
        snakemake.config["electricity"],
        Nyears,
    )

    other_costs = prepare_costs(
        # snakemake.input.tech_costs,
        "../technology-data/outputs/costs_2030.csv",
        snakemake.config["costs"],
        1.,
    )

    generation_emission, daccs_removal, beccs_removal = (
        get_power_generation_emission(
            snakemake.input.fes_table_2023,
            fes,
            year,
        )
    )

    logger.info(f"Emission from Electricity Generation: {np.around(generation_emission, decimals=2)} MtCO2")
    logger.info(f"Direct Air Capture Removal: {np.around(daccs_removal, decimals=2)} MtCO2")
    logger.info(f"Removal through Carbon Capture Biomass: {np.around(beccs_removal, decimals=2)} MtCO2")

    net_change_atmospheric_co2 = generation_emission - daccs_removal - beccs_removal

    logger.info(("\n Net change in atmospheric CO2: ",
        f"{np.around(net_change_atmospheric_co2, decimals=2)} MtCO2"))

    logger.info("Scaling conventional generators to match FES.")
    scale_generation_capacity(n, snakemake.input.capacity_constraints)

    logger.info("Converting conventional generators to links.")
    convert_generators_to_links(n, other_costs)

    logger.warning("Implemented unelegant clean-up of generator marginal costs.")
    if 'GB0 Z11 coal' in n.generators_t.marginal_cost.columns:
        n.generators_t.marginal_cost.drop(columns=[
            'GB0 Z11 coal', 'GB0 Z10 coal', 'GB0 Z8 coal'
            ], inplace=True)

    logger.info("Adding GB CO2 tracking.")
    add_gb_co2_tracking(n, net_change_atmospheric_co2)
     
    logger.info("Adding direct air capture.")
    add_dac(n, other_costs)

    logger.info("Adding gas CCS generation.")
    add_gas_ccs(n, other_costs)

    logger.info("Adding biogas to the system")
    add_biogas(n, other_costs)
    
    logger.info("Adding heat pump load.")
    add_heat_pump_load(
        n,
        snakemake.input["heat_demand"],
        snakemake.input["cop_air_total"],
        snakemake.input["energy_totals"],
        snakemake.input["heat_profile"],
        snakemake.wildcards.fes,
        snakemake.wildcards.year,
    )

    logger.info("Adding BEV load.")
    add_bev(n,
        snakemake.config["sector"],
    )

    logger.info("Adding battery storage.")
    add_batteries(n)

    logger.info("Adding transmission limit.")
    set_line_s_max_pu(n, snakemake.config["lines"]["s_max_pu"])

    flexopts = snakemake.wildcards.flexopts.split("-")

    assert sum(list(map(lambda x: x in {"ss", "reg", "bev", "heat"}, flexopts))) == len(flexopts), (
        "flexopts must be a combination of 'ss', 'reg', 'bev', 'heat', currently is {}".format(
            snakemake.wildcards.flexopts
        )
    )

    logger.info(f"Using Flexibility Options: {flexopts}")

    if "reg" in flexopts:
        add_flexibility(n, "regular")
    
    if "ss" in flexopts:
        add_flexibility(n, "winter")
    
    # import sys
    # sys.exit()

    for o in opts:
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break

    for o in opts:
        m = re.match(r"^\d+seg$", o, re.IGNORECASE)
        if m is not None:
            solver_name = snakemake.config["solving"]["solver"]["name"]
            n = apply_time_segmentation(n, m.group(0)[:-3], solver_name)
            break

    for o in opts:
        if "Co2L" in o:
            m = re.findall("[0-9]*\.?[0-9]+$", o)
            if len(m) > 0:
                co2limit = float(m[0]) * snakemake.config["electricity"]["co2base"]
                add_co2limit(n, co2limit, Nyears)
                logger.info("Setting CO2 limit according to wildcard value.")
            else:
                add_co2limit(n, snakemake.config["electricity"]["co2limit"], Nyears)
                logger.info("Setting CO2 limit according to config value.")
            break

    for o in opts:
        if "CH4L" in o:
            m = re.findall("[0-9]*\.?[0-9]+$", o)
            if len(m) > 0:
                limit = float(m[0]) * 1e6
                add_gaslimit(n, limit, Nyears)
                logger.info("Setting gas usage limit according to wildcard value.")
            else:
                add_gaslimit(n, snakemake.config["electricity"].get("gaslimit"), Nyears)
                logger.info("Setting gas usage limit according to config value.")
            break

    for o in opts:
        oo = o.split("+")
        suptechs = map(lambda c: c.split("-", 2)[0], n.carriers.index)
        if oo[0].startswith(tuple(suptechs)):
            carrier = oo[0]
            # handles only p_nom_max as stores and lines have no potentials
            attr_lookup = {"p": "p_nom_max", "c": "capital_cost", "m": "marginal_cost"}
            attr = attr_lookup[oo[1][0]]
            factor = float(oo[1][1:])
            if carrier == "AC":  # lines do not have carrier
                n.lines[attr] *= factor
            else:
                comps = {"Generator", "Link", "StorageUnit", "Store"}
                for c in n.iterate_components(comps):
                    sel = c.df.carrier.str.contains(carrier)
                    c.df.loc[sel, attr] *= factor

    for o in opts:
        if "Ep" in o:
            m = re.findall("[0-9]*\.?[0-9]+$", o)
            if len(m) > 0:
                logger.info("Setting emission prices according to wildcard value.")
                add_emission_prices(n, dict(co2=float(m[0])))
            else:
                logger.info("Setting emission prices according to config value.")
                add_emission_prices(n, snakemake.config["costs"]["emission_prices"])
            break
        if "ept" in o:
            logger.info("Setting time dependent emission prices according spot market price")
            add_emission_prices_t(n) 

    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    set_transmission_limit(n, ll_type, factor, costs, Nyears)

    set_line_nom_max(
        n,
        s_nom_max_set=snakemake.config["lines"].get("s_nom_max,", np.inf),
        p_nom_max_set=snakemake.config["links"].get("p_nom_max,", np.inf),
    )

    if "ATK" in opts:
        enforce_autarky(n)
    elif "ATKc" in opts:
        enforce_autarky(n, only_crossborder=True)


    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

    n.links.to_csv("links.csv")
    n.lines.to_csv("lines.csv")
    n.buses.to_csv("buses.csv")
    n.generators.to_csv("generators.csv")
    n.stores.to_csv("stores.csv")
    n.storage_units.to_csv("storage_units.csv")
    n.loads.to_csv("loads.csv")
    n.links_t.marginal_cost.to_csv("mcosts_links.csv")
    n.generators_t.marginal_cost.to_csv("mcosts_generators.csv")
    n.stores_t.marginal_cost.to_csv("mcosts_stores.csv")
    n.loads_t.p_set.to_csv("loads_pset.csv")

    n.export_to_netcdf(snakemake.output[0])
