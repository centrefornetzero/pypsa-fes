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
from add_electricity import load_costs, update_transmission_costs
from prepare_sector_network import cycling_shift, prepare_costs
from _fes_helpers import (
    get_data_point,
    scenario_mapper,
    get_gb_total_number_cars,
    get_gb_total_transport_demand,
    get_smart_charge_v2g,
    get_power_generation_emission,
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

    for fes_gen, target in generation_mapper.items():
        index = gb_gen[gb_gen.carrier.isin(target)].index

        if not fes_gen in constraints.index:
            n.generators.drop(index, inplace=True)
            continue

        adapted = n.generators.loc[index, "p_nom"].copy()
        adapted *= constraints.loc[fes_gen] / adapted.sum() 

        n.generators.loc[index, "p_nom"] = adapted
        n.generators.loc[index, "p_nom_min"] = adapted


def convert_generators_to_links(n, costs):

    logger.info("Converting generators to links")

    gb_generation = n.generators.loc[n.generators.bus.str.contains("GB")]
    conventionals = {"CCGT": "gas",
                     "OCGT": "gas",
                     "coal": "coal",
                     "lignite": "lignite"}

    for generator, carrier in conventionals.items():

        gens = gb_generation.loc[gb_generation.carrier == generator]

        if gens.empty:
            print(f"nothing to do for carrier {generator}")
            continue

        if not f"GB_{carrier}_bus" in n.buses.index:

            print(f"Adding bus GB_{carrier}_bus")

            n.add("Bus",
                f"GB_{carrier}_bus",
                carrier=carrier)

            n.add("Generator",
                  f"GB_{carrier}",
                  bus=f"GB_{carrier}_bus",
                  carrier=carrier,
                  p_nom_extendable=True,
                  marginal_cost=costs.at[carrier, "fuel"],
                  )

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

    year = int(snakemake.wildcards.planning_horizons)

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
        snakemake.wildcards.fes_scenario,
        year)
    
    electric_share = bev_cars / gb_cars_2050
    logger.info(f"EV share: {electric_share*100}%")

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
            snakemake.wildcards.fes_scenario,
            year)
        
        if v2g_share > 0.:
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

        if smart_share > 0.:
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


def add_gb_co2_tracking(n):
    # can also be negative
    n.add(
        "Bus",
        "gb co2 atmosphere",
        carrier="co2",
        unit="t_co2",
    )

    n.add(
        "Store",
        "gb co2 atmosphere",
        e_nom_extendable=True,
        carrier="co2",
        bus="gb co2 atmosphere",
    )

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
        carrier="co2",
        bus="gb co2 stored",
    )


def add_dac(n, costs):

    gb_buses = n.buses.loc[n.buses.index.str.contains("GB")]
    gb_buses = gb_buses.loc[gb_buses.carrier == "AC"]

    logger.info("Adding direct air capture")
    logger.warning("Neglecting heat demand of direct air capture")

    logger.warning("Changed sign of DAC efficiency to be positive")
    efficiency2 = (
        costs.at["direct air capture", "electricity-input"]
        + costs.at["direct air capture", "compression-electricity-input"]
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
        efficiency=1.0,
        efficiency2=efficiency2,
        p_nom_extendable=True,
        lifetime=costs.at["direct air capture", "lifetime"],
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

    # scale_generation_capacity(n, snakemake.input.capacity_constraints)
    # convert_generators_to_links(n, other_costs)

    # logger.warning("Implemented unelegant clean-up of generator marginal costs")
    # if 'GB0 Z11 coal' in n.generators_t.marginal_cost.columns:
    #     n.generators_t.marginal_cost.drop(columns=[
    #         'GB0 Z11 coal', 'GB0 Z11 coal', 'GB0 Z8 coal'
    #         ], inplace=True)

    # add_gb_co2_tracking(n)
    # add_dac(n, other_costs)

    # set_line_s_max_pu(n, snakemake.config["lines"]["s_max_pu"])

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

    """
    add_heat_pump_load(
        n, 
        snakemake.input["heat_demand"],
        snakemake.input["cop_air_total"],
        snakemake.input["energy_totals"],
        snakemake.input["heat_profile"],
        snakemake.wildcards.fes_scenario,
        snakemake.wildcards.planning_horizons,
    )

    add_bev(n,
        snakemake.config["sector"],
    )
    """

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

    n.links.to_csv("links_before_saving_network.csv")
    n.buses.to_csv("buses_before_saving_network.csv")
    n.generators.to_csv("generators_before_saving_network.csv")
    n.stores.to_csv("stores_before_saving_network.csv")
    n.loads.to_csv("stores_before_saving_network.csv")
    n.links_t.marginal_cost.to_csv("links_before_saving_network.csv")
    n.generators_t.marginal_cost.to_csv("generators_before_saving_network.csv")
    n.stores_t.marginal_cost.to_csv("stores_before_saving_network.csv")
    n.loads_t.p_set.to_csv("stores_before_saving_network.csv")

    n.export_to_netcdf(snakemake.output[0])
