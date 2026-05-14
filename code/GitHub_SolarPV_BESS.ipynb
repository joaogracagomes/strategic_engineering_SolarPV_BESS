# %% [markdown]
# **Paper:** *A Strategic Engineering Analysis of Flexible and Robust Deployment of Solar PV Power Plants Coupled with Battery Energy Storage Systems*
#
# **Authors:** João Graça Gomes, Michel-Alexandre Cardin, Billy Wu — Dyson School of Design Engineering, Imperial College London
#
# ---
#
# ### Overview
#
# This notebook implements the three-step methodology described in the paper:
#
# | Step | Model Type | Description | Paper Reference |
# |------|-----------|-------------|-----------------|
# | **Step 1** | Deterministic | Baseline LCOE/LCOS for fixed & phased PV+BESS deployment schedules | Eqs. (1)–(16) |
# | **Step 2** | Stochastic | Monte Carlo simulation with uncertain costs and curtailment (EA) | Eqs. (17)–(20) |
# | **Step 3** | Flexible (ROA) | Adaptive capacity expansion via EA-triggered decision rules | Eqs. (21)–(30) |
#
# The core idea follows the Strategic Engineering framework (Cardin, 2014):
# rather than optimising for a single forecast, the model evaluates **flexible deployment strategies**
# that can adapt as uncertainty unfolds — quantifying the **Value of Flexibility (VOF)** through
# distributional LCOE comparisons across fixed, phased, and adaptive configurations.
#
# ### Case Study
# - **Location:** Alentejo, Portugal (high solar irradiance region)
# - **System:** 100 MW Solar PV + 60 MW / 60 MWh LFP BESS
# - **Horizon:** 20-year operational lifetime
# - **Uncertainty:** Effective Availability (EA), technology costs, O&M costs

# %% [markdown]
# ## 1. Library Imports

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.stats as stats
import seaborn as sns

from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

# %% [markdown]
# ## 2. Project-Level Constants and Techno-Economic Parameters
#
# | Parameter | Value | Meaning |
# |-----------|-------|---------|
# | `POWER_CAPACITY` | 100 MW | Nameplate PV plant capacity (θ_max) |
# | `USD_TO_EUR` | 0.9243 | USD → EUR conversion factor |
# | `PROJECT_LIFETIME` | 20 years | Operational horizon T |
# | `DISCOUNT_RATE` (λ) | 3% | Nominal discount rate — Eq. (1) |
# | `DEGRADATION_RATE_PV` (ρ^PV) | 0.52%/yr | Annual PV module degradation — Eq. (7) |
# | `LR_EXOGENOUS` | 20% | Exogenous learning rate LR_exo — Eq. (8) |
# | `LR_ENDOGENOUS` | 3% | Endogenous learning rate LR_end — Eq. (8) |

# %%
# =============================================================================
# PROJECT-LEVEL CONSTANTS
# =============================================================================

POWER_CAPACITY = 100              # [MW] Nameplate PV plant capacity (θ_max)
USD_TO_EUR = 0.9243               # USD-to-EUR conversion factor
PROJECT_LIFETIME = 20             # [years] Planning horizon T — Eq. (1)
DISCOUNT_RATE = 0.03              # [-] Nominal discount rate λ — Eq. (1)

# Sensitivity sweep vectors
EOS_EXPONENTS = [0.6, 0.7, 0.8, 0.9, 1.0]    # α — Eq. (6)
EOS_BASELINE = 1.0                              # α = 1 → no scale advantage

# Solar PV technical parameters
DEGRADATION_RATE_PV = 0.0052      # [-] Annual PV module degradation ρ^PV — Eq. (7)
DEGRADATION_RATES_SWEEP = [0.0052, 0.01, 0.03, 0.05, 0.07, 0.09]

CONVERSION_POWER = 10             # Scaling factor: raw solar data → MW-scale generation

# Learning rates — Eqs. (8), (14)–(15)
LR_EXOGENOUS = 0.20               # LR_exo: 20% cost reduction per doubling of global capacity
LR_EXOGENOUS_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
LR_ENDOGENOUS = 0.03              # LR_end: 3% cost reduction per doubling of local capacity

# Site-specific costs (Portugal)
FEASIBILITY_COST_EUR_MW = 35_000   # [€/MW] Engineering design & preliminary studies
LAND_RENTING_EUR_MW = 2342         # [€/MW] Annual land rental cost (fixed tilt)
SALVAGE_PV_EUR_MW = 25_760         # [€/MW] End-of-life decommissioning — C_dec,PV
FLEXIBILITY_PREMIUM = 0.05         # 5% upfront cost for flexible design enablers

# BESS technical parameters
BATTERY_EFFICIENCY = 0.83          # [-] Round-trip efficiency η_char × η_dis
DOD = 0.80                         # [-] Depth of Discharge
DEGRADATION_RATE_BESS = 0.001      # [-] Annual BESS degradation rate ρ^BESS
DEGRADATION_RATES_BESS_SWEEP = [0.0052, 0.01, 0.03, 0.05, 0.07, 0.09]
TOTAL_BATTERY_CYCLES = 4500        # Total cycles over battery lifetime
CYCLES_PER_YEAR = TOTAL_BATTERY_CYCLES / PROJECT_LIFETIME  # 225 cycles/year
SALVAGE_BATTERY_EUR_KWH = 2.39     # [€/kWh] End-of-life salvage — C_dec,BESS

# BESS learning rate
LR_BESS = 0.10                     # 10% baseline BESS learning rate
LR_BESS_SWEEP = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

# Inverter cost (was missing in original — CRITICAL FIX)
INVERTER_COST_PER_WATT = 0.05 * USD_TO_EUR  # [€/W] — update with your actual value

# %% [markdown]
# ## 3. PV Module, Tracking System, and Operational Cost Data
#
# The model evaluates six PV module technologies and three mounting/tracking configurations.
# Costs are sourced from industry benchmarks and converted to EUR.

# %%
# =============================================================================
# PV MODULE COST DATA — C_mod in Eq. (5)
# =============================================================================
module_data = {
    "Standard_AI_BSF": {"cost_per_watt": 0.39 * USD_TO_EUR},
    "PERC":            {"cost_per_watt": 0.38 * USD_TO_EUR},
    "Multi_perc":      {"cost_per_watt": 0.37 * USD_TO_EUR},
    "Bifacial_pert":   {"cost_per_watt": 0.40 * USD_TO_EUR},
    "Bifacial_SHJ":    {"cost_per_watt": 0.40 * USD_TO_EUR},
    "IBC":             {"cost_per_watt": 0.41 * USD_TO_EUR},
}

# =============================================================================
# TRACKING SYSTEM COSTS — C_trac in Eq. (5)
# =============================================================================
tracking_costs = {
    "Fixed": {
        "below5MW":   0.70 * USD_TO_EUR,
        "below10MW":  0.60 * USD_TO_EUR,
        "below50MW":  0.49 * USD_TO_EUR,
        "below100MW": 0.58 * USD_TO_EUR,
    },
    "OneAxis": {
        "below5MW":   (1.22 - 0.34) * USD_TO_EUR,
        "below10MW":  (1.13 - 0.34) * USD_TO_EUR,
        "below50MW":  (0.98 - 0.34) * USD_TO_EUR,
        "below100MW": (0.89 - 0.34) * USD_TO_EUR,
    },
    "DualAxis": {
        "below5MW":   (1.13 + 1.015 - 0.34) * USD_TO_EUR,
        "below10MW":  (1.03 + 1.015 - 0.34) * USD_TO_EUR,
        "below50MW":  (0.91 + 1.015 - 0.34) * USD_TO_EUR,
        "below100MW": (0.83 + 1.015 - 0.34) * USD_TO_EUR,
    },
}

# Module efficiency multipliers — η_mod in Eq. (7)
module_efficiency = {
    "Standard_AI_BSF": 1.00,
    "PERC":            1.03,
    "Multi_perc":      1.02,
    "Bifacial_pert":   1.05,
    "Bifacial_SHJ":    1.10,
    "IBC":             1.12,
}

# Tracker energy yield multipliers — A_trac in Eq. (7)
tracker_multipliers = {
    "Fixed":    1.000,
    "OneAxis":  1.096,
    "DualAxis": 1.395,
}

# PV OpEx — OM_fix,PV in Eq. (5)
opex_per_kw_per_year = {
    "Fixed":    17,
    "OneAxis":  36,
    "DualAxis": 101,
}

# %% [markdown]
# ## 4. BESS Cost Data
#
# BESS costs loaded from external Excel workbook (PNNL LFP data).
# Cost components map to paper notation: DC_C, DC_BOS, SI, Peq, CC, G_int.
#
# > **Note:** Update `BATTERY_DATA_PATH` to your local file.

# %%
# =============================================================================
# BESS COST DATA — Eq. (9)
# =============================================================================

BATTERY_DATA_PATH = r"/Battery/Cost_battery.xlsx"  # UPDATE THIS PATH

grid_ratio = 0.2

# --- Load cost tiers ---
Cost_1MW_df = pd.read_excel(BATTERY_DATA_PATH, sheet_name="LFB", usecols="A:K", header=0, index_col=0, nrows=10)
Cost_1MW_df = Cost_1MW_df.apply(pd.to_numeric, errors="coerce")
cost_dict = Cost_1MW_df.to_dict()

Cost_10MW_df = pd.read_excel(BATTERY_DATA_PATH, sheet_name="LFB", usecols="A:K", header=11, index_col=0, nrows=10)
Cost_10MW_df = Cost_10MW_df.apply(pd.to_numeric, errors="coerce")
cost_dict_10 = Cost_10MW_df.to_dict()

Cost_100MW_df = pd.read_excel(BATTERY_DATA_PATH, sheet_name="LFB", usecols="A:K", header=22, index_col=0, nrows=10)
Cost_100MW_df = Cost_100MW_df.apply(pd.to_numeric, errors="coerce")
cost_dict_100 = Cost_100MW_df.to_dict()

print("BESS cost data loaded successfully")
print("Available storage durations:", list(cost_dict.keys()))

# %% [markdown]
# ## 5. Solar Irradiation Data
#
# Hourly solar power generation profiles for Alentejo, Portugal (2014–2023).
# Source: [EU PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/tools.html)
#
# > **Note:** Update `SOLAR_DIR` to your local directory.

# %%
# =============================================================================
# SOLAR IRRADIATION DATA — Historical profiles (2014–2023)
# =============================================================================

SOLAR_DIR = r"/path/to/solar/data"  # UPDATE THIS PATH

def load_solar_profile(filepath):
    """Load a single year's hourly solar profile and return scaled MW array."""
    df = pd.read_excel(filepath, sheet_name='Folha1', usecols='B:D', nrows=8759)
    df['HourIndex'] = df['Day'] * 24 + df['Hour']
    power = df.set_index('HourIndex')['Power'].reindex(range(1, 8761), fill_value=0).tolist()
    return np.array([CONVERSION_POWER * p for p in power])

# Load historical profiles
solar_profiles = {}
for year in range(2014, 2024):
    filepath = os.path.join(SOLAR_DIR, f"Solarpower_{year}.xls")
    solar_profiles[year] = load_solar_profile(filepath)
    print(f"  {year}: {solar_profiles[year].sum():.0f} MWh")

# Generate synthetic profiles for projection period (2024–2033)
# Using bootstrap resampling from historical data with small perturbation
historical_list = [solar_profiles[y] for y in range(2014, 2022)]
power_array = np.array(historical_list)
average_power = np.mean(power_array, axis=0)

np.random.seed(42)  # Reproducibility
for year in range(2024, 2034):
    base_idx = np.random.randint(0, len(historical_list))
    noise = np.random.normal(1.0, 0.02, 8760)
    solar_profiles[year] = historical_list[base_idx] * noise

# Assign profiles to planning horizon years (1–24)
P_yearly = {}
for i, year in enumerate(range(2014, 2034)):
    P_yearly[i] = solar_profiles[year]
# Extended years use last available profile
for i in range(20, 25):
    P_yearly[i] = solar_profiles[2033]

# Annual generation totals for LCOE denominator
P_profiles = [solar_profiles[y].sum() for y in range(2014, 2034)]

Total_generation = sum(P_profiles)
print(f"\nTotal generation over {PROJECT_LIFETIME}-year lifetime: {Total_generation:,.0f} MWh")

# %% [markdown]
# ## 6. Deployment Scenarios and Helper Functions

# %%
# =============================================================================
# PV DEPLOYMENT SCHEDULES — χ = (θ₀, θ₁, ..., θ_T)
# =============================================================================
installation_scenarios = {
    "100 MW in Year 0":                        [100] + [0] * PROJECT_LIFETIME,
    "50 MW in Year 0 and 50 MW in Year 1":     [50, 50] + [0] * (PROJECT_LIFETIME - 1),
    "50 MW first year, 25 MW next two years":  [50, 25, 25] + [0] * (PROJECT_LIFETIME - 2),
    "75 MW first year, 25 MW next year":       [75, 25] + [0] * (PROJECT_LIFETIME - 1),
    "25 MW first year, 25 MW next years":      [25, 25, 25, 25] + [0] * (PROJECT_LIFETIME - 3),
}

# =============================================================================
# BESS DEPLOYMENT SCHEDULES — (δ₀, δ₁, ..., δ_T)
# =============================================================================
storage_durations = [1]  # [hours] — S_hours in Eq. (9)

battery_installation_scenarios = {
    "60 MW in Year 0":                          [60] + [0] * PROJECT_LIFETIME,
    "50 MW in Year 0 and 10 MW in Year 1":      [50, 10] + [0] * (PROJECT_LIFETIME - 1),
    "20 MW first year, 20 MW next 2 years":     [20, 20, 20] + [0] * (PROJECT_LIFETIME - 2),
    "30 MW first year, 30 MW next 1 year":      [30, 30] + [0] * (PROJECT_LIFETIME - 1),
    "10 MW first year, 20 MW next 1 year, 30":  [10, 20, 30] + [0] * (PROJECT_LIFETIME - 2),
}

def determine_cost_category(installed_capacity_mw: float) -> str:
    """Map installed capacity to cost bracket for tracker/BESS pricing."""
    if installed_capacity_mw < 6:
        return "below5MW"
    elif installed_capacity_mw < 11:
        return "below10MW"
    elif installed_capacity_mw < 51:
        return "below50MW"
    else:
        return "below100MW"

def select_bess_cost_tier(capacity_mw: float) -> dict:
    """Select BESS cost dictionary based on installed capacity tier."""
    if capacity_mw < 10:
        return cost_dict
    elif capacity_mw < 61:
        return cost_dict_10
    else:
        return cost_dict_100

def sample_bess_cost(battery_cost_dict: dict, storage_hours: int,
                     capacity_kwh: float, capacity_kw: float,
                     lf: float, perturbation: bool = False) -> float:
    """
    Compute total BESS cost for a given capacity and cost tier.
    
    If perturbation=True, applies ±10% triangular distribution (Step 2/3).
    If perturbation=False, uses nominal values (Step 1).
    """
    key = f"{storage_hours} hours"
    
    energy_components = [
        "DC Storage Block ($/kWh)", "DC Storage BOS ($/kWh)",
        "Systems Integration ($/kWh)", "EPC ($/kWh)", "Project Development ($/kWh)"
    ]
    power_components = [
        "Power Equipment ($/kW)", "CC ($/kW)", "Grid Integration ($/kW)"
    ]
    
    if perturbation:
        energy_cost = sum(
            np.random.triangular(0.9 * battery_cost_dict[key][k],
                                 battery_cost_dict[key][k],
                                 1.1 * battery_cost_dict[key][k])
            for k in energy_components
        )
        power_cost = sum(
            np.random.triangular(0.9 * battery_cost_dict[key][k],
                                 battery_cost_dict[key][k],
                                 1.1 * battery_cost_dict[key][k])
            for k in power_components
        )
    else:
        energy_cost = sum(battery_cost_dict[key][k] for k in energy_components)
        power_cost = sum(battery_cost_dict[key][k] for k in power_components)
    
    return (capacity_kwh * energy_cost + capacity_kw * power_cost) * USD_TO_EUR * lf

def sample_bess_opex(battery_cost_dict: dict, storage_hours: int,
                     capacity_kw: float, perturbation: bool = False) -> float:
    """Compute annual BESS operation cost."""
    key = f"{storage_hours} hours"
    opex_rate = battery_cost_dict[key]["Operation Cost ($/kW-year)"]
    if perturbation:
        opex_rate = np.random.triangular(0.9 * opex_rate, opex_rate, 1.1 * opex_rate)
    return opex_rate * capacity_kw

# %% [markdown]
# ---
# ## Step 1 — Deterministic LCOE/LCOS Evaluation
#
# **Paper reference: Section 2.1.1, Eqs. (1)–(16)**
#
# Evaluates all combinations of:
# - 5 PV schedules × 5 BESS schedules × 6 modules × 3 trackers = **450 configurations**
#
# LCOE = Σ C_t(θ,δ)/(1+λ)^t ÷ Σ E_t/(1+λ)^t — Eq. (1)

# %%
# =============================================================================
# STEP 1: DETERMINISTIC LCOE / LCOS COMPUTATION
# =============================================================================

results_step1 = []

for scen_batt_name, sched_batt in battery_installation_scenarios.items():
    total_batt_mw = sum(sched_batt)

    for storage_hours in storage_durations:
        sh = int(round(storage_hours))

        for scen_pv_name, sched_pv in installation_scenarios.items():
            for tracker, tracker_cost_dict in tracking_costs.items():
                tracker_mult = tracker_multipliers[tracker]
                tracker_opex = opex_per_kw_per_year[tracker]

                for module, mod_info in module_data.items():
                    last_inst_yr = max(i for i, c in enumerate(sched_pv) if c > 0)
                    ext_lifetime = last_inst_yr + 21
                    mod_eff = module_efficiency[module]

                    # Financial accumulators
                    total_solar_capex = 0.0
                    total_battery_capex = 0.0
                    npv_num_lcoe = 0.0
                    npv_den_lcoe = 0.0
                    npv_num_lcos = 0.0
                    npv_den_lcos = 0.0
                    cum_pv_mw = 0
                    cum_batt_mw = 0
                    salvage_lcoe = 0.0
                    salvage_lcos = 0.0
                    decomm_pv = 0
                    decomm_batt = 0

                    for year in range(ext_lifetime):
                        df = 1 / ((1 + DISCOUNT_RATE) ** year) if year > 0 else 1.0

                        inst_pv = sched_pv[year] if year < len(sched_pv) else 0
                        inst_batt = sched_batt[year] if year < len(sched_batt) else 0

                        # Decommission at end of 20-year life
                        if year >= 20:
                            decomm_pv = sched_pv[year - 20] if (year - 20) < len(sched_pv) else 0
                            cum_pv_mw -= decomm_pv
                            decomm_batt = sched_batt[year - 20] if (year - 20) < len(sched_batt) else 0
                            cum_batt_mw -= decomm_batt

                        cum_pv_mw += inst_pv
                        cum_batt_mw += inst_batt

                        # --- PV CAPEX — Eq. (5) with learning — Eq. (8) ---
                        if inst_pv > 0:
                            cost_cat = determine_cost_category(inst_pv)
                            tracker_cpw = tracker_cost_dict[cost_cat]
                            lf_tech = (1 - LR_EXOGENOUS) ** max(0, year - 1)
                            lf_exp = (1 - LR_ENDOGENOUS) ** max(0, year - 1)

                            cap_w = inst_pv * 1e6
                            mod_cost = cap_w * mod_info["cost_per_watt"] * lf_tech * lf_exp
                            trk_cost = cap_w * tracker_cpw
                            inv_cost = cap_w * INVERTER_COST_PER_WATT
                            total_solar_capex += (mod_cost + trk_cost + inv_cost) * df

                        # --- BESS CAPEX — Eq. (9) ---
                        batt_dict = select_bess_cost_tier(inst_batt)
                        lf_bess = (1 - LR_BESS) ** max(0, year - 1)
                        batt_kwh = inst_batt * 1e3 * sh
                        batt_kw = inst_batt * 1e3
                        batt_cost = sample_bess_cost(batt_dict, sh, batt_kwh, batt_kw, lf_bess, perturbation=False)
                        total_battery_capex += batt_cost * df

                        # --- Energy generation — Eq. (7) ---
                        gen_pv = 0.0
                        gen_batt = 0.0

                        if year > 0:
                            for prev_yr in range(max(0, year - 19), year + 1):
                                act_pv = sched_pv[prev_yr] if prev_yr < len(sched_pv) else 0
                                deg = (1 - DEGRADATION_RATE_PV) ** (year - prev_yr)
                                gen_pv += (
                                    P_profiles[min(year, len(P_profiles) - 1)]
                                    * act_pv / POWER_CAPACITY
                                    * tracker_mult * mod_eff * deg
                                )

                            for prev_yr in range(max(0, year - 19), year + 1):
                                act_batt = sched_batt[prev_yr] if prev_yr < len(sched_batt) else 0
                                deg_b = (1 - DEGRADATION_RATE_BESS) ** (year - prev_yr)
                                gen_batt += (
                                    act_batt * sh * BATTERY_EFFICIENCY
                                    * DOD * deg_b * CYCLES_PER_YEAR
                                )

                            # OpEx
                            batt_opex = sample_bess_opex(batt_dict, sh, batt_kw, perturbation=False)
                            npv_num_lcoe += (
                                tracker_opex * cum_pv_mw * 1e3 * df
                                + LAND_RENTING_EUR_MW / 20 * inst_pv * df
                                + batt_opex * df
                            )
                            npv_num_lcos += batt_opex * df
                            npv_den_lcoe += (gen_pv + gen_batt) * df
                            npv_den_lcos += gen_batt * df

                        # Decommissioning
                        if decomm_pv > 0:
                            salvage_lcoe += SALVAGE_PV_EUR_MW * decomm_pv * df
                        if decomm_batt > 0:
                            salvage_lcos += SALVAGE_BATTERY_EUR_KWH * 1e3 * sh * decomm_batt * df

                    # --- Final LCOE and LCOS — Eqs. (1) and (16) ---
                    salvage_total = salvage_lcoe + salvage_lcos
                    lcoe = (total_solar_capex + total_battery_capex + npv_num_lcoe + salvage_total) / npv_den_lcoe if npv_den_lcoe != 0 else float("inf")
                    lcos = (total_battery_capex + npv_num_lcos + salvage_lcos) / npv_den_lcos if npv_den_lcos != 0 else float("inf")

                    results_step1.append({
                        "Scenario PV": scen_pv_name,
                        "Scenario Battery": scen_batt_name,
                        "Tracker Type": tracker,
                        "Module Type": module,
                        "Battery Capacity (MW)": total_batt_mw,
                        "Storage Hours": sh,
                        "Solar CAPEX (EUR)": total_solar_capex,
                        "Battery CAPEX (EUR)": total_battery_capex,
                        "Total CAPEX (EUR)": total_solar_capex + total_battery_capex,
                        "LCOE (EUR/MWh)": lcoe,
                        "LCOS (EUR/MWh)": lcos,
                        "Total Solar Capacity (MW)": sum(sched_pv),
                        "Total Battery Capacity (MWh)": total_batt_mw * sh,
                    })

results_step1_df = pd.DataFrame(results_step1)
print(f"Step 1 complete: {len(results_step1_df)} configurations evaluated")

# %% [markdown]
# ### Step 1 — Results Summary

# %%
lcoe_stats = results_step1_df['LCOE (EUR/MWh)'].agg(['mean', 'max', 'min'])
lcos_stats = results_step1_df['LCOS (EUR/MWh)'].agg(['mean', 'max', 'min'])
print("LCOE Statistics:\n", lcoe_stats)
print("\nLCOS Statistics:\n", lcos_stats)

lowest = results_step1_df.nsmallest(4, 'LCOE (EUR/MWh)')
print("\nTOP 4 LOWEST LCOE CONFIGURATIONS:")
print(lowest.to_string(max_colwidth=30, index=False))

# Filter for Fixed + IBC (best technology combo for subsequent steps)
fixed_ibc_step1 = results_step1_df[
    (results_step1_df['Tracker Type'] == 'Fixed') &
    (results_step1_df['Module Type'] == 'IBC')
]
print(f"\nFixed+IBC LCOE range: {fixed_ibc_step1['LCOE (EUR/MWh)'].min():.1f} – {fixed_ibc_step1['LCOE (EUR/MWh)'].max():.1f} EUR/MWh")

# %% [markdown]
# ---
# ## Step 2 — Stochastic LCOE/LCOS via Monte Carlo Simulation
#
# **Paper reference: Section 2.1.2, Eqs. (17)–(20)**
#
# Extends the deterministic model with uncertainty in:
# 1. **Effective Availability (EA)**: availability-reduction factor ρ_t sampled from
#    triangular distribution. EA_t = 1 - ρ_t. Note: `sample_rho()` returns ρ_t.
# 2. **Technology costs**: ±10–20% perturbation via triangular distributions (Υ_PV, Υ_BESS).

# %%
# =============================================================================
# STEP 2: STOCHASTIC MONTE CARLO SIMULATION
# =============================================================================

print("Starting Step 2...")

# --- Availability-reduction factor bounds (ρ_t, NOT EA directly) ---
# EA_t = 1 - ρ_t. Derived from GBM simulations of SAIDI, failures, curtailment.
YEARS_OP = np.arange(1, 26)

RHO_BASELINE = np.array([
    0.0118, 0.0043, 0.0033, 0.0089, 0.0036, 0.0034, 0.0031, 0.0030,
    0.0032, 0.0027, 0.0025, 0.0024, 0.0024, 0.0022, 0.0021, 0.0023,
    0.0018, 0.0016, 0.0015, 0.0011, 0.0010, 0.0007, 0.0007, 0.0007, 0.0007
])

RHO_HIGHER = np.array([
    0.1191, 0.1413, 0.1604, 0.1430, 0.1489, 0.1437, 0.1633, 0.1558,
    0.1704, 0.1731, 0.2125, 0.2427, 0.2277, 0.2038, 0.2058, 0.2200,
    0.2274, 0.2301, 0.2402, 0.2403, 0.2339, 0.2402, 0.2402, 0.2402, 0.2402
])

RHO_MEAN = (RHO_BASELINE + RHO_HIGHER) / 2

f_rho_low = interp1d(YEARS_OP, RHO_BASELINE, kind='linear', fill_value="extrapolate")
f_rho_mid = interp1d(YEARS_OP, RHO_MEAN, kind='linear', fill_value="extrapolate")
f_rho_high = interp1d(YEARS_OP, RHO_HIGHER, kind='linear', fill_value="extrapolate")


def sample_rho(year_of_operation: int) -> float:
    """
    Sample availability-reduction factor ρ_t from a triangular distribution.
    Returns ρ_t (NOT EA). Effective availability: EA_t = 1 - ρ_t.
    """
    lo = float(f_rho_low(year_of_operation))
    mid = float(f_rho_mid(year_of_operation))
    hi = float(f_rho_high(year_of_operation))
    return np.random.triangular(lo, mid, hi)


MC_ITERATIONS_STEP2 = 2000

results_step2 = []

for sim in range(MC_ITERATIONS_STEP2):
    if sim % 500 == 0:
        print(f"  Step 2: simulation {sim}/{MC_ITERATIONS_STEP2}")

    for scen_batt_name, sched_batt in battery_installation_scenarios.items():
        total_batt_mw = sum(sched_batt)

        for sh in [int(round(s)) for s in storage_durations]:
            for scen_pv_name, sched_pv in installation_scenarios.items():
                for tracker, tracker_cost_dict in tracking_costs.items():
                    tracker_mult = tracker_multipliers[tracker]

                    # Stochastic OpEx — ±20% perturbation
                    tracker_opex = np.random.triangular(
                        0.8 * opex_per_kw_per_year[tracker],
                        opex_per_kw_per_year[tracker],
                        1.2 * opex_per_kw_per_year[tracker]
                    )

                    for module, mod_info in module_data.items():
                        last_inst_yr = max(i for i, c in enumerate(sched_pv) if c > 0)
                        ext_lifetime = last_inst_yr + 21
                        mod_eff = module_efficiency[module]

                        total_solar_capex = 0.0
                        total_battery_capex = 0.0
                        npv_num_lcoe = 0.0
                        npv_den_lcoe = 0.0
                        npv_num_lcos = 0.0
                        npv_den_lcos = 0.0
                        cum_pv_mw = 0
                        cum_batt_mw = 0
                        salvage_lcoe = 0.0
                        salvage_lcos = 0.0
                        decomm_pv = 0
                        decomm_batt = 0

                        for year in range(ext_lifetime):
                            df = 1 / ((1 + DISCOUNT_RATE) ** year) if year > 0 else 1.0
                            inst_pv = sched_pv[year] if year < len(sched_pv) else 0
                            inst_batt = sched_batt[year] if year < len(sched_batt) else 0

                            if year >= 20:
                                decomm_pv = sched_pv[year - 20] if (year - 20) < len(sched_pv) else 0
                                cum_pv_mw -= decomm_pv
                                decomm_batt = sched_batt[year - 20] if (year - 20) < len(sched_batt) else 0
                                cum_batt_mw -= decomm_batt

                            cum_pv_mw += inst_pv
                            cum_batt_mw += inst_batt

                            # PV CAPEX with stochastic cost perturbation
                            if inst_pv > 0:
                                cost_cat = determine_cost_category(inst_pv)
                                trk_cpw = np.random.triangular(
                                    0.8 * tracker_cost_dict[cost_cat],
                                    tracker_cost_dict[cost_cat],
                                    1.2 * tracker_cost_dict[cost_cat]
                                )
                                mod_cpw = np.random.triangular(
                                    0.8 * mod_info["cost_per_watt"],
                                    mod_info["cost_per_watt"],
                                    1.2 * mod_info["cost_per_watt"]
                                )
                                lf_tech = (1 - LR_EXOGENOUS) ** max(0, year - 1)
                                lf_exp = (1 - LR_ENDOGENOUS) ** max(0, year - 1)
                                cap_w = inst_pv * 1e6
                                total_solar_capex += (
                                    cap_w * mod_cpw * lf_tech + cap_w * trk_cpw * lf_exp
                                ) * df

                            # BESS CAPEX with stochastic perturbation
                            batt_dict = select_bess_cost_tier(inst_batt)
                            lf_bess = (1 - LR_BESS) ** max(0, year - 1)
                            batt_kwh = inst_batt * 1e3 * sh
                            batt_kw = inst_batt * 1e3
                            total_battery_capex += sample_bess_cost(
                                batt_dict, sh, batt_kwh, batt_kw, lf_bess, perturbation=True
                            ) * df

                            # Energy generation with stochastic EA
                            if year > 0:
                                # CRITICAL FIX: sample ρ ONCE per year, OUTSIDE vintage loop
                                rho_t = sample_rho(min(year + 1, 25))

                                raw_gen = 0.0
                                for prev_yr in range(max(0, year - 19), year + 1):
                                    act_pv = sched_pv[prev_yr] if prev_yr < len(sched_pv) else 0
                                    deg = (1 - DEGRADATION_RATE_PV) ** (year - prev_yr)
                                    raw_gen += (
                                        P_profiles[min(year, len(P_profiles) - 1)]
                                        * act_pv / POWER_CAPACITY
                                        * tracker_mult * mod_eff * deg
                                    )
                                gen_pv = raw_gen * (1 - rho_t)  # EA applied once

                                gen_batt = 0.0
                                for prev_yr in range(max(0, year - 19), year + 1):
                                    act_b = sched_batt[prev_yr] if prev_yr < len(sched_batt) else 0
                                    deg_b = (1 - DEGRADATION_RATE_BESS) ** (year - prev_yr)
                                    gen_batt += act_b * sh * BATTERY_EFFICIENCY * DOD * deg_b * CYCLES_PER_YEAR

                                batt_opex = sample_bess_opex(batt_dict, sh, batt_kw, perturbation=True)
                                npv_num_lcoe += (
                                    tracker_opex * cum_pv_mw * 1e3 * df
                                    + LAND_RENTING_EUR_MW / 20 * inst_pv * df
                                    + batt_opex * df
                                )
                                npv_num_lcos += batt_opex * df
                                npv_den_lcoe += (gen_pv + gen_batt) * df
                                npv_den_lcos += gen_batt * df

                            if decomm_pv > 0:
                                salvage_lcoe += SALVAGE_PV_EUR_MW * decomm_pv * df
                            if decomm_batt > 0:
                                salvage_lcos += np.random.triangular(
                                    0.9 * SALVAGE_BATTERY_EUR_KWH,
                                    SALVAGE_BATTERY_EUR_KWH,
                                    1.1 * SALVAGE_BATTERY_EUR_KWH
                                ) * 1e3 * sh * decomm_batt * df

                        salvage_total = salvage_lcoe + salvage_lcos
                        lcoe = (total_solar_capex + total_battery_capex + npv_num_lcoe + salvage_total) / npv_den_lcoe if npv_den_lcoe != 0 else float("inf")
                        lcos = (total_battery_capex + npv_num_lcos + salvage_lcos) / npv_den_lcos if npv_den_lcos != 0 else float("inf")

                        results_step2.append({
                            "Scenario PV": scen_pv_name,
                            "Scenario Battery": scen_batt_name,
                            "Tracker Type": tracker,
                            "Module Type": module,
                            "Battery Capacity (MW)": total_batt_mw,
                            "Storage Hours": sh,
                            "Solar CAPEX (EUR)": total_solar_capex,
                            "Battery CAPEX (EUR)": total_battery_capex,
                            "Total CAPEX (EUR)": total_solar_capex + total_battery_capex,
                            "LCOE (EUR/MWh)": lcoe,
                            "LCOS (EUR/MWh)": lcos,
                            "Total Solar Capacity (MW)": sum(sched_pv),
                            "Total Battery Capacity (MWh)": total_batt_mw * sh,
                        })

results_step2_df = pd.DataFrame(results_step2)
print(f"Step 2 complete: {len(results_step2_df)} realisations")

# Filter for Fixed + IBC
Step2_fixed_ibc_df = results_step2_df[
    (results_step2_df['Tracker Type'] == 'Fixed') &
    (results_step2_df['Module Type'] == 'IBC')
].copy()

# %% [markdown]
# ---
# ## Step 3 — Flexible Capacity Expansion with Decision Rules
#
# **Paper reference: Section 2.1.3, Eqs. (21)–(30)**
#
# Adaptive decision rules expand PV/BESS capacity contingent on observed EA.
#
# **PV rule F_t** (expand when curtailment LOW — grid can absorb more):
# - ρ_t < 5% → θ_H = 40 MW | 5–8% → θ_M = 25 MW | 8–10% → θ_L = 10 MW | ≥10% → 0
#
# **BESS rule H_t** (INVERTED — expand when curtailment HIGH):
# - ρ_t > 15% → δ_H = 40 MW | 10–15% → δ_M = 20 MW | 5–10% → δ_L = 10 MW | <5% → 0
#
# VOF = E[LCOE_rigid] − E[LCOE_flexible]

# %%
# =============================================================================
# STEP 3: FLEXIBLE DEPLOYMENT WITH EA-TRIGGERED DECISION RULES
# =============================================================================

print("Starting Step 3...")

# Configuration space
INITIAL_PV_CAPACITIES = [25, 50, 75, 100]     # [MW]
INITIAL_BESS_CAPACITIES = [0, 10, 20, 30, 40, 60]  # [MW]
MAX_EXPANSION_YEAR = 5
MAX_PV_MW = 100
MAX_BESS_MW = 60


def decide_pv_expansion(rho_t: float, cum_pv: float, year: int) -> int:
    """PV expansion decision rule F_t — Eq. (28)."""
    if year >= MAX_EXPANSION_YEAR:
        return 0
    remaining = MAX_PV_MW - cum_pv
    if rho_t < 0.05:
        return min(40, int(remaining)) if remaining > 0 else 0
    elif 0.05 <= rho_t < 0.08:
        return 25 if (cum_pv + 25) <= MAX_PV_MW else 0
    elif 0.08 <= rho_t < 0.10:
        return 10 if (cum_pv + 10) <= MAX_PV_MW else 0
    else:
        return 0


def decide_bess_expansion(rho_t: float, cum_bess: float, year: int) -> int:
    """BESS expansion decision rule H_t — Eq. (29). Logic INVERTED vs PV."""
    if year >= MAX_EXPANSION_YEAR:
        return 0
    remaining = MAX_BESS_MW - cum_bess
    if rho_t > 0.15:
        return min(40, int(remaining)) if remaining > 0 else 0
    elif 0.10 < rho_t <= 0.15:
        return 20 if (cum_bess + 30) <= MAX_BESS_MW else 0
    elif 0.05 <= rho_t <= 0.10:
        return 10 if (cum_bess + 10) <= MAX_BESS_MW else 0
    else:
        return 0


def generate_flexible_schedule(initial_cap, rho_values, decide_fn):
    """Generate path-dependent deployment schedule from decision rules."""
    schedule = [initial_cap]
    cum = initial_cap
    for year in range(1, PROJECT_LIFETIME):
        rho = rho_values[year] if year < len(rho_values) else rho_values[-1]
        addition = decide_fn(rho, cum, year)
        schedule.append(addition)
        cum += addition
    return schedule


# --- Monte Carlo simulation ---
MC_ITERATIONS_STEP3 = 2000
results_step3 = []

for sim in range(MC_ITERATIONS_STEP3):
    if sim % 500 == 0:
        print(f"  Step 3: simulation {sim}/{MC_ITERATIONS_STEP3}")

    rho_values = [sample_rho(yr) for yr in range(PROJECT_LIFETIME)]

    for init_pv in INITIAL_PV_CAPACITIES:
        sched_pv = generate_flexible_schedule(init_pv, rho_values, decide_pv_expansion)

        for init_bess in INITIAL_BESS_CAPACITIES:
            sched_bess = generate_flexible_schedule(init_bess, rho_values, decide_bess_expansion)

            for sh in [int(round(s)) for s in storage_durations]:
                for tracker, tracker_cost_dict in tracking_costs.items():
                    tracker_mult = tracker_multipliers[tracker]
                    tracker_opex = np.random.triangular(
                        0.8 * opex_per_kw_per_year[tracker],
                        opex_per_kw_per_year[tracker],
                        1.2 * opex_per_kw_per_year[tracker]
                    )

                    for module, mod_info in module_data.items():
                        last_inst_yr = max([i for i, c in enumerate(sched_pv) if c > 0], default=0)
                        ext_lifetime = last_inst_yr + 21
                        mod_eff = module_efficiency[module]

                        total_solar_capex = 0.0
                        total_battery_capex = 0.0
                        npv_num_lcoe = 0.0
                        npv_den_lcoe = 0.0
                        npv_num_lcos = 0.0
                        npv_den_lcos = 0.0
                        cum_pv_mw = 0
                        cum_batt_mw = 0
                        salvage_lcoe = 0.0
                        salvage_lcos = 0.0
                        max_pv_reached = 0
                        max_bess_reached = 0
                        decomm_pv = 0
                        decomm_batt = 0
                        eos_alpha = EOS_BASELINE  # Parameterised EoS exponent

                        for year in range(ext_lifetime):
                            df = 1 / ((1 + DISCOUNT_RATE) ** year) if year > 0 else 1.0
                            inst_pv = sched_pv[year] if year < len(sched_pv) else 0
                            inst_batt = sched_bess[year] if year < len(sched_bess) else 0

                            if year >= 20:
                                decomm_pv = sched_pv[year - 20] if (year - 20) < len(sched_pv) else 0
                                cum_pv_mw -= decomm_pv
                                decomm_batt = sched_bess[year - 20] if (year - 20) < len(sched_bess) else 0
                                cum_batt_mw -= decomm_batt

                            cum_pv_mw += inst_pv
                            cum_batt_mw += inst_batt
                            max_pv_reached = max(max_pv_reached, cum_pv_mw)
                            max_bess_reached = max(max_bess_reached, cum_batt_mw)

                            # PV CAPEX with EoS
                            if inst_pv > 0:
                                cost_cat = determine_cost_category(inst_pv)
                                trk_cpw = np.random.triangular(
                                    0.8 * tracker_cost_dict[cost_cat],
                                    tracker_cost_dict[cost_cat],
                                    1.2 * tracker_cost_dict[cost_cat]
                                )
                                mod_cpw = np.random.triangular(
                                    0.8 * mod_info["cost_per_watt"],
                                    mod_info["cost_per_watt"],
                                    1.2 * mod_info["cost_per_watt"]
                                )
                                lf_tech = (1 - LR_EXOGENOUS) ** max(0, year - 1)
                                lf_exp = (1 - LR_ENDOGENOUS) ** max(0, year - 1)
                                cap_w = inst_pv * 1e6
                                eos_factor = cum_pv_mw ** (eos_alpha - 1) if cum_pv_mw > 0 else 1.0
                                total_solar_capex += (
                                    cap_w * mod_cpw * lf_tech * eos_factor
                                    + cap_w * trk_cpw * lf_exp * eos_factor
                                ) * df

                            # BESS CAPEX with EoS
                            batt_dict = select_bess_cost_tier(inst_batt)
                            lf_bess = (1 - LR_BESS) ** max(0, year - 1)
                            batt_kwh = inst_batt * 1e3 * sh
                            batt_kw = inst_batt * 1e3
                            batt_cost = sample_bess_cost(batt_dict, sh, batt_kwh, batt_kw, lf_bess, perturbation=True)
                            eos_bess = cum_batt_mw ** (eos_alpha - 1) if cum_batt_mw > 0 else 1.0
                            total_battery_capex += batt_cost * eos_bess * df

                            # Energy generation
                            if year > 0:
                                # CRITICAL FIX: sample ρ ONCE per year
                                rho_t = sample_rho(min(year + 1, 25))

                                raw_gen = 0.0
                                for prev_yr in range(max(0, year - 19), year + 1):
                                    act_pv = sched_pv[prev_yr] if prev_yr < len(sched_pv) else 0
                                    deg = (1 - DEGRADATION_RATE_PV) ** (year - prev_yr)
                                    raw_gen += (
                                        P_profiles[min(year, len(P_profiles) - 1)]
                                        * act_pv / POWER_CAPACITY
                                        * tracker_mult * mod_eff * deg
                                    )
                                gen_pv = raw_gen * (1 - rho_t)

                                gen_batt = 0.0
                                for prev_yr in range(max(0, year - 19), year + 1):
                                    act_b = sched_bess[prev_yr] if prev_yr < len(sched_bess) else 0
                                    deg_b = (1 - DEGRADATION_RATE_BESS) ** (year - prev_yr)
                                    gen_batt += act_b * sh * BATTERY_EFFICIENCY * DOD * deg_b * CYCLES_PER_YEAR

                                batt_opex = sample_bess_opex(batt_dict, sh, batt_kw, perturbation=True)
                                npv_num_lcoe += (
                                    tracker_opex * cum_pv_mw * 1e3 * df
                                    + LAND_RENTING_EUR_MW / 20 * inst_pv * df
                                    + batt_opex * df
                                )
                                npv_num_lcos += batt_opex * df
                                npv_den_lcoe += (gen_pv + gen_batt) * df
                                npv_den_lcos += gen_batt * df

                            if decomm_pv > 0:
                                salvage_lcoe += SALVAGE_PV_EUR_MW * decomm_pv * df
                            if decomm_batt > 0:
                                salvage_lcos += np.random.triangular(
                                    0.9 * SALVAGE_BATTERY_EUR_KWH,
                                    SALVAGE_BATTERY_EUR_KWH,
                                    1.1 * SALVAGE_BATTERY_EUR_KWH
                                ) * 1e3 * sh * decomm_batt * df

                        salvage_total = salvage_lcoe + salvage_lcos
                        lcoe = (total_solar_capex + total_battery_capex + npv_num_lcoe + salvage_total) / npv_den_lcoe if npv_den_lcoe != 0 else float("inf")
                        lcos = (total_battery_capex + npv_num_lcos + salvage_lcos) / npv_den_lcos if npv_den_lcos != 0 else float("inf")

                        results_step3.append({
                            "Initial PV Capacity": init_pv,
                            "Initial BESS Capacity": init_bess,
                            "Tracker Type": tracker,
                            "Module Type": module,
                            "Storage Hours": sh,
                            "Degradation Rate": DEGRADATION_RATE_PV,
                            "Peak PV Capacity (MW)": max_pv_reached,
                            "Peak BESS Capacity (MW)": max_bess_reached,
                            "BESS Capacity (MWh)": cum_batt_mw * sh,
                            "Solar CAPEX (EUR)": total_solar_capex,
                            "Battery CAPEX (EUR)": total_battery_capex,
                            "LCOE (EUR/MWh)": lcoe,
                            "LCOS (EUR/MWh)": lcos,
                            "Total CAPEX (EUR)": total_solar_capex + total_battery_capex,
                            "Simulation": sim,
                        })

results_step3_df = pd.DataFrame(results_step3)
print(f"Step 3 complete: {len(results_step3_df)} realisations")

# Filter for Fixed + IBC
ibc_fixed_step3 = results_step3_df[
    (results_step3_df['Module Type'] == 'IBC') &
    (results_step3_df['Tracker Type'] == 'Fixed')
].copy()

# Export
ibc_fixed_step3[["Initial PV Capacity", "Initial BESS Capacity", "Storage Hours",
                  "Degradation Rate", "Simulation", "LCOE (EUR/MWh)", "LCOS (EUR/MWh)"]
].to_csv("lcoe_lcos_ibc_fixed_montecarlo.csv", index=False)
print("Exported Step 3 results to lcoe_lcos_ibc_fixed_montecarlo.csv")

# %% [markdown]
# ### Step 3 — Heatmap of P50 LCOE across initial capacities

# %%
for dr in sorted(results_step3_df['Degradation Rate'].unique()):
    df_dr = ibc_fixed_step3[ibc_fixed_step3['Degradation Rate'] == dr]

    lcoe_p50 = (
        df_dr.groupby(['Initial PV Capacity', 'Initial BESS Capacity'])['LCOE (EUR/MWh)']
        .median()
        .reset_index()
    )
    heatmap_data = lcoe_p50.pivot(
        index='Initial PV Capacity',
        columns='Initial BESS Capacity',
        values='LCOE (EUR/MWh)'
    ).iloc[::-1]

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", cmap="crest",
        linewidths=0.5, cbar_kws={'label': 'P50 LCOE (€/MWh)'},
        annot_kws={'size': 14}
    )
    plt.title(f"P50 LCOE — Fixed+IBC (PV degradation = {dr:.2%})", fontsize=16)
    plt.xlabel("Initial BESS Capacity (MW/MWh)", fontsize=14)
    plt.ylabel("Initial PV Capacity (MW)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"heatmap_p50_lcoe_{dr:.4f}.png", dpi=300, bbox_inches='tight')
    plt.show()

print("All steps complete.")
