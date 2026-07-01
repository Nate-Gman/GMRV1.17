#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 GmansRun V1.17  --  HOHEV-Rotary Gen 4 Standalone Digital Twin
================================================================================

A 100% standalone monolith that builds, animates and *drives* the entire
HOHEV-Rotary electric vehicle (EV) powertrain in real time, mechanically, to
scale, in a single Python file.

Two modes (toggle with TAB):

  1. PREVIEW MODE  (loads first -- the "preview display before drive load-in")
     A navigable 3D view of the WHOLE engine assembly. Every part spins and
     works mechanically: the 8-chamber rotary combustion ring fires, the
     internal wet clutch engages toward 1:1, the central shaft drives the
     free-floating tungsten kinetic flywheel, the concentric transmission
     ring layers bind outward by selected gear, and the axial-flux generator +
     small angled "receival" output gear spin up.
     Orbit / zoom / pan with the mouse. Part callouts label every component.

  2. DRIVE MODE  (the "drive load-in version")
     Drive the EV like any car game over a procedurally hilly route. Throttle,
     brake, coast, cruise-control, force engine bursts, and watch the live
     energy economy: battery SOC window, supercapacitor buffer, flywheel
     wind-up on downhill regen, engine-off time, and a live MPG counter aiming
     at the 1000+ MPG concept target. The 3D engine keeps running in an inset.

Dependencies:  numpy, pygame   (both standard installs; no GPU/CAD libs needed)
Run:           python3 GmansRunV1.17.py

Controls are printed at startup and shown on-screen (press  H  for help,
press  I  for the full informational specification panel).

This file is the single source of truth for the concept. Every dimension below
is held in DIMS in millimetres and rendered to scale.
================================================================================
"""

import math
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame


# =============================================================================
# SECTION 1 -- ENGINEERING SPECIFICATION (to scale, millimetres / SI)
# =============================================================================

DIMS = {
    # --- Rotary combustion ring (the "circular piston") -------------------
    "ring_outer_d_mm":      760.0,   # outer diameter
    "ring_inner_d_mm":      520.0,   # hollow inner diameter (holds transmission)
    "ring_thickness_mm":    165.0,   # axial thickness (kept thin on purpose)
    "chambers":             8,       # combustion paddles, equally spaced 45 deg
    "chamber_depth_mm":     70.0,    # concave paddle radial depth
    "chamber_width_mm":     120.0,   # paddle tangential width

    # --- Central free-floating tungsten kinetic flywheel ------------------
    "flywheel_d_mm":        460.0,   # sits in the hollow center, front face
    "flywheel_thick_mm":    90.0,
    "flywheel_mass_kg":     58.0,    # tungsten-composite rim

    # --- Wet clutch stack (1st clutch: ring -> shaft) ---------------------
    "clutch1_d_mm":         480.0,
    "clutch1_thick_mm":     22.0,

    # --- Central shaft (the "Core") ---------------------------------------
    "shaft_d_mm":           70.0,
    "shaft_len_mm":         360.0,

    # --- Transmission gear-increase core (concentric slip rings) ----------
    "trans_rings":          6,       # concentric "gears"; shifter slides up
    "trans_outer_d_mm":     500.0,
    "trans_inner_d_mm":     90.0,
    "trans_thick_mm":       60.0,

    # --- Axial-flux generator ---------------------------------------------
    "gen_d_mm":             300.0,
    "gen_thick_mm":         110.0,

    # --- Toothed "receival" output gear (shifts to ring edge -> generator) -
    "out_gear_d_mm":         34.0,
    "out_gear_thick_mm":     34.0,
    "out_angle_deg":        14.0,    # the "unavoidable angle loss" pivot

    # --- Exhaust turbocharger + slow automatic supercharger ---------------
    "turbo_turbine_d_mm":   105.0,
    "turbo_comp_d_mm":       94.0,
    "turbo_len_mm":          62.0,
    "turbo_runner_d_mm":     28.0,
    "supercharger_d_mm":     92.0,
    "supercharger_len_mm":   78.0,
    "supercharger_pulley_d_mm": 72.0,

    # --- Regenerative closed-loop steam heat-recovery (compact, to scale) --
    "coolring_gap_mm":       26.0,   # cooling ring standoff from the piston OD
    "coolring_thick_mm":     20.0,   # cooling-ring / heat-trap jacket wall
    "boiler_d_mm":           96.0,   # compact steam boiler drum (to scale)
    "boiler_len_mm":        120.0,
    "steam_chamber_d_mm":    44.0,   # dual alternating valve/expansion chambers
    "steam_stage_d_mm":      54.0,   # compound expander stage wheel (x3 stages)
    "steam_gen_d_mm":        86.0,   # compact steam-driven generator
    "steam_gen_thick_mm":    58.0,
    "relief_d_mm":           34.0,   # pressure-release valve / vent turbine
    "shield_span_mm":       150.0,   # geometric heat shield (funnel to boiler)
    "shield_vanes":          7,      # angled reflector vanes aiming the heat
    "recov2_turbine_d_mm":   78.0,   # 2nd-stage exhaust recovery turbine
    "recov2_gen_d_mm":       64.0,
    "wheel_d_mm":           560.0,   # road wheel (regen generator in the hub)
    "wheel_gen_d_mm":       300.0,   # rail-ring regen generator disc
    "wheel_width_mm":       150.0,

    # --- Passenger pop-out pedal-assist trickle generators ---------------
    "seats":                  4,     # one pop-out pedal generator per seat
    "pedal_crank_d_mm":      92.0,   # pedal crank / flywheel
    "pedal_gen_d_mm":        70.0,   # mini trickle generator

    # --- Solar roof panel + windshield concentrating lens ----------------
    "solar_roof_len_mm":   2200.0,   # PV roof panel length (over the cabin/boat-tail)
    "solar_roof_width_mm": 1300.0,   # PV roof panel width
    "windshield_lens_d_mm": 520.0,   # auto-adjusting Fresnel / magnifying lens aperture
    "lens_receiver_d_mm":    72.0,   # concentrated-sun receiver drum on the boiler

    # --- Through-body flow-through duct (rear-wake / pressure-drag kill) ---
    "flow_duct_width_mm":   620.0,   # wide but flat straight-through intake duct
    "flow_duct_height_mm":   95.0,   # thin, for ground clearance / structural integrity
    "flow_duct_len_mm":    2850.0,   # runs nearly the full vehicle length
}

# Passenger pedal-assist: pushing the pop-out foot pedals spins a mini generator
# that trickle-charges the battery, like pedaling a bike. Sustained human output.
PEDAL_WATTS_PER_SEAT = 110.0     # ~a fit cyclist's steady output per seat
PEDAL_RAMP = 2.5                 # how fast pedaling spins up/down

# Real-life MPG estimate (steady flat cruise, textbook ROAD-LOAD physics -- NOT
# the arcade drive economy). Gasoline is only ever burned to make electricity, so
# fuel-MPG is set by how little road power is left after the free sources (pedals,
# and on a real route, regen). Road power = (aero + rolling) x speed:
#   aero  = 1/2 * rho * Cd * A * v^2   -> power grows as v^3 (why speed hurts)
#   roll  = Crr * m * g                -> power grows only as v^1
FUEL_KWH_PER_GAL   = 33.6    # gasoline LHV energy per US gallon (thermal)
FUEL_TO_WHEEL_EFF  = 0.465   # optimizer-scaled kinetic->electrical chain (mech .955 x gen
#                              .975 x gear .988) + steam recovery + .97 rail -> ~46.5% fuel->wheel
OCCUPANT_KG        = 80.0    # mass added per occupant beyond the driver
MPG_SPEEDS_MPH     = [5, 10, 25, 50, 80]
MPG_PLOT_CAP       = 2000.0  # chart clamps above this into the top band
# When free EXTERNAL power (solar) fully covers the road load, no fuel burns, so the
# fuel-MPG is unbounded for that point. We do NOT call that "infinite MPG" (it reads as
# magic) -- it is SOLAR-SUSTAINED cruise, shown as a capped ">cap" figure labelled SOLAR.
MPG_MAX_DISPLAY    = 9999.0
PEDAL_HELP_FRAC    = 0.05    # below this share of road load, pedals retract

# Firing / RPM behaviour
FIRING_IDLE_RPM   = 320.0     # always-spinning warm idle when vehicle moving
FIRING_GEN_RPM    = 400.0     # optimized burst RPM (drive uses 8 combustions/rev)
DRIVE_COMBUSTIONS = 8         # corrected optimizer sweet spot for range-extender bursts
FLYWHEEL_MAX_RPM  = 20000.0   # magnetic-bearing safe ceiling
OMEGA_VIEW        = 2.513     # preview rotation rate at 1.0x (~2.5 s / rev, viewable)
INJECTOR_ANGLE    = math.pi / 2.0   # injector fixed at the top of the rim
MAX_PISTON_RPM    = 4500.0    # piston RPM at full power / 8 combustions per rev
VIEW_ROT_SCALE    = 0.03      # slows real RPMs to a viewable on-screen spin rate
VISUAL_DETAIL     = 1.45      # higher mesh resolution for preview inspection
SUPERCHARGER_BOOST_GAIN = 0.22
SUPERCHARGER_RAMP_RATE  = 0.42

# Closed-loop working fluids for the boiler. A LOW-BOILING-POINT fluid flashes to
# vapor at low temperature, so it makes usable pressure from low-grade waste heat
# that water cannot -- more total steam power over a drive. Crucially, the useful
# fluids ALSO do not freeze in winter, so they are strictly better than water,
# which freezes at 0 C (a cold-weather liability that can crack the loop).
#   boil_c  = useful-vapor onset      full_c = pressure saturation
#   max_bar = closed-loop PSI ceiling gen_eff/capacity = power throughput
#   freeze_c = freezing point         heat_ok_c = heat the fluid tolerates
#   service_kwh = recovered energy the charge lasts before a fluid change
# AMMONIA is the default: highest pressure/power AND it withstands the most heat
# and the longest service cycle -- it takes the abuse of a long-duration loop.
# (Compared head-to-head in the sim; every non-freezing fluid beats WATER.)
FLUIDS = {
    "AMMONIA":  {"boil_c": 26.0, "full_c": 92.0,  "max_bar": 26.0,   # high-PSI, strong
                 "gen_eff": 0.245, "capacity_kw": 88.0, "freeze_c": -78.0,
                 "heat_ok_c": 230.0, "service_kwh": 4200.0},
    "R245FA":   {"boil_c": 30.0, "full_c": 96.0,  "max_bar": 21.0,   # ORC refrigerant
                 "gen_eff": 0.230, "capacity_kw": 80.0, "freeze_c": -103.0,
                 "heat_ok_c": 175.0, "service_kwh": 2600.0},
    "METHANOL": {"boil_c": 55.0, "full_c": 135.0, "max_bar": 15.0,   # simple antifreeze
                 "gen_eff": 0.205, "capacity_kw": 72.0, "freeze_c": -98.0,
                 "heat_ok_c": 195.0, "service_kwh": 2000.0},
    "WATER":    {"boil_c": 82.0, "full_c": 165.0, "max_bar": 14.0,   # freezes at 0 C!
                 "gen_eff": 0.190, "capacity_kw": 66.0, "freeze_c": 0.0,
                 "heat_ok_c": 240.0, "service_kwh": 1500.0},
}
FLUID_ORDER = ["AMMONIA", "R245FA", "METHANOL", "WATER"]
DEFAULT_FLUID = "AMMONIA"

# Shared heat-recovery thresholds (the physical heat model lives in PHYS).
# HEAT-TRAPPING closed-loop steam-regeneration: the cooling ring around the piston
# absorbs heat, a geometric heat SHIELD funnels it to the compact boiler, and the
# vapour pressure is milked by a DUAL-CHAMBER, MULTI-STAGE (compound) expander that
# reuses the pressure internally until it is weak. The aim is to KEEP heat and make
# electricity from pressure differentials -- not to throw heat away.
THERM = {
    "ambient_c":       35.0,   # ambient
    "orc_eff":         0.14,   # legacy key (kept for back-compat)
    "steam_c":         80.0,   # legacy fallback (fluid boil_c overrides)
    "steam_full_c":   165.0,   # legacy fallback (fluid full_c overrides)
    "heat_trap_frac":   0.90,  # cooling ring + shield keep this much of the heat
    "second_stage_frac": 0.28, # 2nd turbine recovers this of leftover exhaust
    "exhaust_to_boiler_frac": 0.78,  # remaining exhaust heat routed to the boiler
    # Geometric heat-SHIELD directs block heat into the boiler focus (replaces the
    # old heat-absorbing metal ingots -- geometry, not mass, aims the heat).
    "shield_ua_kw_c":   0.48,  # block -> boiler directed-heat coupling
    "boiler_mass_kj_c": 26.0,  # compact boiler + working-fluid thermal inertia
    "boiler_leak_kw_c": 0.010, # small ambient loss from the compact boiler
    "condition_ua_kw_c": 0.55, # last-resort active thermal conditioning (regulate)
    # Internal multi-stage pressure REUSE (compound expander) + dual alternating
    # valve chambers: pressure is milked until weak with no dead stroke.
    "compound_stages":  4,     # internal pressure-reuse stages (more sustained gain)
    "stage_gain":       0.12,  # extra recovery each internal reuse stage adds
    "dual_chamber_gain": 1.18, # two chambers alternating = near-continuous duty
    "recovery_eff_cap": 0.58,  # ceiling on heat->electric (optimizer-scaled, Carnot-bounded)
    # PHASE-CHANGE MATERIAL (PCM) heat BANK wrapped around the boiler. It soaks up
    # excess heat while the engine fires (latent melt), then releases it slowly for
    # many minutes after shutdown -- so the steam loop keeps making electricity long
    # after the burner stops, stretching engine-off time.
    "pcm_kj_c":         120.0, # PCM sensible + latent thermal inertia (large on purpose)
    "pcm_couple_kw_c":   0.09, # boiler <-> PCM heat-exchange coupling
    "pcm_leak_kw_c":     0.004,# tiny ambient loss from the insulated PCM bank
}

# First-order engineering physics model. This is not CFD/FEM, but it now uses
# measured geometry, ideal-gas charge mass, fuel LHV, hydraulic clamp force,
# clutch slip work, gear mesh efficiency, coolant UA and overheat derate.
PHYS = {
    "ambient_pa":             101325.0,
    "intake_temp_k":             330.0,
    "air_r":                     287.05,
    "gasoline_lhv_j_kg":    44_000_000.0,
    "gasoline_kg_per_gal":         2.75,
    "target_afr":                 18.5,
    "volumetric_eff":              0.78,  # redesigned ports + boost: better breathing
    "trapped_volume_factor":       0.12,  # actual sealed charge pocket fraction
    "seal_eff":                    0.88,  # tighter apex/face seals: less blow-by loss
    "indicated_eff":               0.44,  # higher-compression, cleaner burn (thermal-limited)
    "mechanical_eff":             0.955,  # magnetic-bearing low-friction (optimizer -> ceiling)
    "generator_eff":              0.975,  # axial-flux PM: KINETIC->ELECTRICAL near ceiling
    "gear_mesh_eff":              0.988,  # gearless rail coupling (optimizer -> ceiling)
    "turbo_pressure_ratio":        1.18,
    "air_psi_optional":            0.0,   # optional internal chamber assist, off by default
    "air_psi_pressure_ratio":      0.06,
    "hydraulic_max_bar":          85.0,
    "main_clutch_area_m2":         0.018,
    "ring_bind_area_m2":           0.026,
    "friction_mu_wet":             0.080,
    "main_clutch_radius_m":        0.190,
    "ring_bind_radius_m":          0.170,
    "bearing_drag_kw":             0.16,  # magnetic / low-drag bearings ("no drag")
    "friction_kw_cap":           160.0,
    "heat_to_block_frac":          0.42,
    "exhaust_recovery_frac":       0.26,
    "thermal_mass_kj_c":          85.0,
    "coolant_ua_kw_c":             0.050,
    "coolant_flow_kw_c":           0.380,
    "overheat_c":                125.0,
    "shutdown_c":                155.0,
    "max_temp_c":                240.0,
}

# Vehicle / physics (lightweight series-hybrid laboratory mule).
# The body was AI shape-optimized (generative-design CFD loop): a teardrop cabin
# with a long boat-tail, faired/covered wheels and a fully sealed flat underbody
# drop Cd to ~0.08 and shrink the frontal area of a narrow tandem cabin. Combined
# with a generative-lattice ultralight structure and low-rolling-resistance tyres,
# this slashes the road load -- the single biggest lever on real-world MPG.
VEH = {
    "curb_mass_kg":    765.0,   # <770 kg: carbon monocoque + STRUCTURAL battery + driver
    "Cd":              0.08,    # AI shape-optimized teardrop + boat-tail, covered wheels
    "frontal_area_m2": 1.58,    # narrower fill-factor tandem cabin (active air dam + underbody)
    "Crr":             0.0028,  # airless metamaterial tyres, active pressure, sealed floor
    "air_density":     1.225,
    "g":               9.81,
    "wheel_radius_m":  0.32,
    "drivetrain_eff":  0.970,   # EM RAIL-RING direct wheel drive (rim IS the rotor, no gearbox,
    #                             SiC inverter) -- optimizer-scaled to the ~97% ceiling
    "max_rail_kw":     320.0,   # 4 direct EM rail-wheel drives (rail-current scaled)
    "max_brake_n":     9000.0,
    "regen_frac":      0.92,    # rail-ring generator + supercap buffer (optimizer-scaled max)
}

# EM RAIL-RING DIRECT WHEEL DRIVE. This is NOT a separate wheel motor or electric
# engine: the stationary electromagnetic rail directly rotates the rim/wheel itself.
# The rail surrounds each wheel like a stator around a rotor, and the WHEEL RIM
# ITSELF is the rotor -- bonded as a single hard-locked piece (no clutch in the
# wheel), riding on a magnetic bearing and extended just under an inch to clear the
# rail. Energising the 96-pole hybrid superconducting + permanent-magnet rail
# magnetically spins/locks the rim with scalable, gearless torque. No gearbox and no
# drive clutch means very low loss (96%+), and the same rail regenerates on braking.
RAIL_DRIVE = {
    "eff":            0.970,    # rail drive + inverter efficiency (matches drivetrain_eff)
    "poles":          96,       # 96-pole hybrid superconducting + PM rail
    "coil_amp_max":   2500.0,   # 0-2500 A ECU-modulated rail current (scales torque)
    "disc_dia_m":     1.00,     # standalone unit main-disc diameter
    "unit_kg":        950.0,    # standalone unit mass (in-wheel version is far lighter)
    "cont_torque_nm": 350000.0, # continuous torque, standalone unit (rail-current scaled)
    "burst_torque_nm":500000.0, # capacitor-boosted burst torque
    "flywheel_rpm":   25000.0,  # inner-disc kinetic-store speed
    "air_gap_mm":     0.15,     # rail-to-rotor precision air gap
}
RAILMOTOR = RAIL_DRIVE  # compatibility alias for existing helper code

# MULTI-LAYER AMBIENT HARVEST -- the compounding, fuel-free trickle that runs even
# with the engine off and the wheels coasting. No single magic part; every joule is
# attacked from several angles at once. Favorable-daylight estimates (watts):
# Sizing OPTIMIZER-SCALED (see optimize_harvest.py): the low-mass, high-value solar
# and TENG/tyre/piezo were pushed toward their physical caps; the heavier suspension
# damper was balanced (bigger flat baseline, trimmed peak) for the best net MPG.
# SOLAR is grounded in panel area x irradiance x efficiency (external energy, honest).
# The MOTION sources (tribo/tyre/piezo) are NAMEPLATE PEAKS only -- at runtime they are
# bounded to a small fraction of the rolling-loss they RECOVER (tyre/body hysteresis
# that is already being dissipated), so they reclaim spent energy, never create it.
AMBIENT_HARVEST_W = {
    "solar":        210.0,   # quantum-dot film over the upper body (~3 m^2 x ~7% effective)
    "solar_roof":   440.0,   # dedicated PV ROOF (~2.0 m^2 x 22% x 1 kW/m^2 = 650 W total solar)
    "suspension":   130.0,   # linear EM dampers -- NAMEPLATE; runtime = bump dissipation only
    "triboelectric": 62.0,   # TENG films on underbody + wheel wells (nameplate)
    "tire":          40.0,   # airless metamaterial tyres with embedded harvesters (nameplate)
    "piezo":         28.0,   # seat + body-panel piezo harvesters (nameplate)
}
# Fraction of the (already-dissipated) rolling-loss power the tyre/tribo/piezo film can
# actually reclaim -- caps the motion harvest to real physics (-> 0 when parked).
HARVEST_ROLL_FRAC = 0.12

# CONCENTRATING SOLAR-TO-BOILER: an auto-adjusting Fresnel / magnifying lens set into
# the windshield tracks the sun and focuses concentrated sunlight onto a small
# secondary receiver on the boiler. In bright hot sun it dumps free HEAT straight
# into the closed-loop fluid -- so the steam expander makes electricity from sunshine
# alone, even parked or cruising engine-off. Regulated (defocused) if it over-pressures.
# PHYSICALLY BOUNDED: a lens only CONCENTRATES the sunlight that lands on its aperture
# (it raises flux DENSITY, not total power), so the thermal power it can deliver is
# capped by aperture_area x irradiance x optical_efficiency -- NOT an arbitrary figure.
SOLAR_IRRADIANCE_W = 1000.0    # full-sun surface irradiance (W/m^2)
_LENS_APERTURE_M2 = math.pi * (DIMS["windshield_lens_d_mm"] * 0.001 / 2.0) ** 2
SOLAR_CONCENTRATOR_KW = _LENS_APERTURE_M2 * SOLAR_IRRADIANCE_W * 0.85 / 1000.0  # ~0.18 kW
SUN = 1.0                      # favourable-daylight sun intensity, 0..1 (bright hot sun)

# ACTIVE MORPHING AERO -- the single biggest MPG lever is aero drag, which grows as
# speed CUBED. Shape-memory / actuated boat-tail, rear diffuser and underbody skirts
# morph at cruise to a lower-drag shape (delaying flow separation, sealing the wheel
# wells, forming a ground-effect tunnel). It retracts under hard accel/brake for
# stability. Modelled as a speed-dependent multiplier on the effective Cd.
AERO = {
    "morph_cd_min_frac": 0.74,   # fully-morphed Cd = 74% of the static Cd (~0.059)
    "morph_onset_mph":   22.0,   # morphing begins to deploy above this
    "morph_full_mph":    58.0,   # fully deployed by highway cruise
}

# SUPER-HYDROPHOBIC + PLASTRON "SLIP" SKIN. A nano-textured super-hydrophobic body
# coating makes rain/humidity BEAD UP and roll straight off instead of forming a
# clinging water film -- so in wet or humid air the car does not drag a skin of
# water with it (a real, measurable drag + weight penalty on ordinary paint). The
# same micro-texture retains a thin trapped AIR LAYER (a "plastron") that behaves
# like a near-frictionless, vacuum-like slip boundary: the passing airflow rides on
# air instead of gripping the paint, cutting turbulent SKIN-FRICTION drag even in
# dry air (the shark-skin riblet effect). Both shrink the effective Cd; the humid
# part grows with how wet the air is.
HYDRO = {
    "dry_frac":   0.050,   # baseline skin-friction cut (riblet + plastron air-slip), any air
    "humid_frac": 0.075,   # EXTRA cut in wet air: no water film to drag (scales with humidity)
}
HUMIDITY = 0.6             # ambient relative humidity 0..1 (higher = more coating benefit)

# PASSIVE PERMANENT-MAGNET (HARD-MAGNETIC) WHEEL BEARINGS. NOT powered electromagnetic
# (active magnetic) bearings -- those burn continuous electrical power in their control
# coils, a parasitic load. These are HARD-MAGNETIC: a Halbach permanent-magnet array
# floats each wheel hub on repelling fields with essentially ZERO rolling friction and
# ZERO electrical draw. Permanent-magnet levitation is statically unstable on its own
# (Earnshaw's theorem), so it is STABILIZED passively -- a diamagnetic (pyrolytic-
# graphite) element plus a low-drag ceramic touchdown race and an eddy-current damper
# pin the remaining axis without powered coils. Result: the bearing-friction slice of
# the rolling/parasitic drag is nearly eliminated, lowering the road load at every
# speed -- pure, always-on MPG.
WHEEL_BEARING = {
    "roll_cut_frac": 0.18,   # fraction of the rolling/bearing drag removed by PM levitation
    "type": "passive Halbach permanent-magnet levitation, diamagnetic-stabilized",
}

# PASSIVE DRAG-AIR CAPTURE & RECOVERY. The hydrophobic skin makes most airflow slip,
# but the little that still grips the body is caught by directional micro-grooves
# (deep shark-skin riblets aligned WITH the airflow) that funnel that boundary-layer
# air into internal ducts driving tiny low-drag (bladeless Tesla-type) turbines. It
# is fully passive -- no pumps or valves -- and it turns a slice of what would be
# WASTED drag energy into electricity, scaling with speed (aero power ~ v^3).
# groove_eff x turbine_eff (~8%) is the fraction of the REMAINING aero drag power
# recovered as electricity. The expanded optimizer trims this after aero/duct gains:
# it is better to avoid drag first than harvest a bigger slice of it.
DRAG_CAPTURE = {
    "groove_eff":  0.20,     # fraction of the remaining drag air the grooves funnel in
    "turbine_eff": 0.15,     # bladeless / small-axial turbine -> ~3% net of the aero drag
    #                          power (energy already lost to the wake; recover-only, honest)
}

# PARKED WINDMILL / DEPLOYABLE TAIL FIN. When the car is parked (or crawling), a rear
# tail fin extends and works as a small VERTICAL-AXIS wind turbine, trickle-charging
# the battery from ambient wind (alongside the solar roof). It retracts while driving
# (where it would just be drag). Output scales with the ambient WIND strength.
WINDMILL = {
    "park_mph": 2.0,         # at/below this the fin is deployed as a windmill
    "max_w":    95.0,        # peak output in strong wind
    "area_m2":  0.65,        # effective capture area
}
WIND = 0.55                  # ambient wind strength 0..1 (parked-windmill charging)

# INERTIAL PENDULUM RIM. A shallow skirt (~2-5 in) rings the base of the body -- the
# lower bumpers, front/back and side to side -- and hangs a row of small PENDULUM
# weights. As the car accelerates, brakes, corners, or its pitch changes on a
# gradient, the effective-gravity vector in the vehicle frame tips, so each hanging
# weight swings back/forth and drives a tiny generator. It is a REGENERATIVE damper
# on the body's own inertial motion: energy that would otherwise be lost as the body
# rocks is turned into charge. It gives the most in stop-go city, cornering, and
# rolling/incline-changing terrain, and ~zero on a dead-straight steady cruise.
PENDULUM = {
    "weights":   28,         # dangling bobs around the lower rim
    "weight_kg": 0.55,       # each pendulum bob mass
    "max_w":      90.0,      # peak harvest (optimizer-sized: lighter bobs beat bigger ones)
    "long_ref":  0.40,       # longitudinal accel (g) that saturates the swing
    "lat_ref":   0.35,       # lateral (cornering) accel (g) that saturates the swing
    "pitch_ref": 0.06,       # incline-change rate (rad/s) that saturates the swing
    # The dangling skirt/weights add AERO DRAG that grows with speed (v^2), so above a
    # threshold the drag it adds costs more than the inertial energy it harvests. It
    # therefore DEPLOYS at low speed (where drag is tiny and stop-go swings are big)
    # and RETRACTS up into the body at high speed (where it would only be a drag).
    "deploy_mph":  38.0,     # fully deployed at/below this
    "retract_mph": 60.0,     # fully retracted (flush) at/above this
    "cd_penalty":  0.012,    # Cd added to the car when the skirt is fully deployed
}

# THROUGH-BODY STRAIGHT FLOW-THROUGH DUCT. A wide, thin, straight hollow duct runs
# from a low front-grill intake, straight through the body, to a rear diffuser. It
# lets high-pressure nose air flow 100% straight through the car and exit into the
# low-pressure WAKE, "filling" the rear vacuum pocket -- attacking PRESSURE (form)
# drag, which dominates above ~50 mph. Modelled as a speed-dependent cut to the
# effective Cd (stronger at speed, where the wake vacuum is biggest).
DUCT = {
    "base":    0.15,   # baseline wake-fill drag reduction (any speed)
    "speed":   0.10,   # EXTRA reduction that ramps in with speed
    "ref_mph": 60.0,   # speed at which the extra reduction saturates
    "blend":   0.78,   # how much of the modelled wake reduction lands on the Cd
}

# WHOLE-VEHICLE dimensions, to real-life scale (millimetres). The CAR VIEW builds the
# entire vehicle from these so the model is dimensionally honest, and the road-load
# physics is cross-checked against them: frontal area ~= width x height x fill factor
# ~ VEH["frontal_area_m2"], and the mass is VEH["curb_mass_kg"]. A narrow tandem
# teardrop: long boat-tail, faired wheels, low + light carbon monocoque.
CAR = {
    "length_mm":       3820.0,   # bumper to bumper
    "width_mm":        1600.0,   # track body width (narrow tandem cabin)
    "height_mm":       1150.0,   # low roofline
    "wheelbase_mm":    2500.0,   # front-to-rear axle
    "track_mm":        1350.0,   # left-to-right wheel centres
    "wheel_d_mm":       620.0,   # road wheel outer diameter
    "wheel_width_mm":   205.0,
    "ground_clear_mm":  120.0,   # sealed flat underbody ride height
    "frontal_fill":       0.86,  # width x height filled by the teardrop cross-section
}

# THERMOELECTRIC GENERATORS (TEG): Seebeck-effect films layered on the hot block and
# exhaust as a SECOND heat-recovery path alongside the steam loop. They make a
# trickle of electricity straight from the block-to-ambient temperature difference,
# and -- like the steam loop -- keep producing after the engine shuts off while the
# block is still hot.
TEG = {
    "kw_per_c": 0.0040,   # electric kW per degC (optimizer-trimmed: mass/heat-sink trade)
    "cap_kw":   1.20,     # ceiling on TEG output
}

# REGENERATIVE STEERING: a small generator on the steering rack/column trickles
# charge whenever the wheel is turned at speed. Minor, but always-on during turns.
STEER_REGEN_W_MAX = 120.0

# DYNAMIC (electromagnetic) SUSPENSION harvest ON REAL ROADS, over and above the
# flat-cruise baseline already counted in AMBIENT_HARVEST_W["suspension"]. Bumps and
# grade shake the linear dampers harder, so the harvest scales with speed, road
# roughness and |grade|.
SUSP_BUMP_W_MAX = 80.0    # peak extra damper harvest (optimizer-trimmed: mass vs benefit)

# Electrical storage. HEADROOM IS KING: hold a narrow 15-52% SOC window so there
# is always empty room to swallow braking + downhill + ambient harvest -- never
# chase a full battery. A big supercapacitor bank takes every peak charge/drain.
ELEC = {
    "batt_kwh":        40.0,    # structural battery integrated into the monocoque
    "soc_min":         0.15,    # never held high; max regen headroom
    "soc_max":         0.52,
    "soc_start":       0.35,    # start mid-window
    "supercap_kwh":    0.40,    # sized for PEAK POWER (fast cycles), not bulk energy: the
    #                             optimizer showed oversizing it just adds mass (~125 kg/kWh)
}

# STORAGE ROUND-TRIP EFFICIENCY. Real storage is NOT lossless -- every joule stored
# then retrieved pays a round-trip. So route each flow through the MOST EFFICIENT
# buffer that can take it: fast charge/drain spikes (braking, engine bursts) go to the
# SUPERCAPS (highest round-trip); bulk/slow storage to the BATTERY; mechanical overflow
# to the FLYWHEEL. The loss is applied on the way IN, so charging then discharging
# NEVER returns more than went in (energy-honest -- no free storage). Sizing the
# supercap bigger keeps more of the fast cycling in the high-eff buffer = lower loss.
STORAGE = {
    "supercap_rt": 0.985,   # supercapacitor round-trip (fast cycles, near-lossless)
    "battery_rt":  0.955,   # structural Li/LFP battery round-trip (bulk)
    "flywheel_rt": 0.905,   # tungsten flywheel on magnetic bearings (mechanical overflow)
}

# Engine control thresholds (keeps engine runtime as low as the route allows).
# The rotary is a RARE high-efficiency burst generator only: it fires when the
# battery + flywheel + live harvesting cannot meet demand, then stops the instant
# EITHER the battery target OR the heat target is reached -- once the block is hot
# the trickle steam-regen keeps harvesting with the engine OFF.
ENGINE_ON_SOC    = 0.20       # fire a burst when buffers run low
ENGINE_OFF_SOC   = 0.42       # stop once the battery reaches the optimized target, OR...
ENGINE_OFF_TEMP_C = 118.0     # ...once the block is hot enough to keep making steam
ENGINE_CRIT_SOC  = 0.13       # safety floor: refire even when hot if this low

# PREDICTIVE CONTROL + DYNAMIC SOC WINDOW. The controller looks a few hundred metres
# AHEAD on the route grade and reshapes the SOC window so there is always room to
# harvest what is coming: before a DESCENT it drops the charge target (empties the
# buffers so the whole downhill can be regenerated instead of dumped as brake heat);
# before a CLIMB it lifts the target (a pre-emptive burst so the climb runs on
# stored charge, keeping the engine off on the hill). Net effect: the burner fires
# less often and captures more free energy -> higher MPG.
PREDICT_LOOKAHEAD_M   = 260.0   # how far ahead the controller reads the grade
PREDICT_SOC_SWING     = 0.10    # max +/- shift of the engine-off SOC target
PREDICT_GRADE_REF     = 0.05    # grade magnitude that saturates the SOC shift

# HIGH-VOLTAGE ELECTRICAL BUS + WIRING (modelled to scale, not a black box). The pack,
# SiC inverter and EM rail-ring drives are wired on an 800 V HV bus. Power is P = V x I,
# so the bus CURRENT is set by the demanded power and the pack voltage, and the copper
# cable (real cross-section + length) drops I^2 x R as heat. inverter_eff x motor_eff =
# 0.985 x 0.985 = 0.970 -- i.e. these ARE the components the aggregate VEH["drivetrain_eff"]
# of 0.97 is made of, now shown explicitly with the cable drop on top.
BUS = {
    "pack_v_nom":   800.0,     # nominal HV pack voltage (series cell string)
    "bus_a_max":    450.0,     # main HV bus current limit (contactor/fuse rating)
    "inverter_eff": 0.985,     # SiC traction inverter efficiency
    "motor_eff":    0.985,     # EM rail-ring motor/generator efficiency
    "wire_mm2":     50.0,      # main HV cable copper cross-section (each conductor)
    "wire_len_m":   4.0,       # round-trip pack <-> inverter cable length
    "cu_ohm_m":     1.72e-8,   # copper resistivity at operating temperature
}
BUS_R_OHM = BUS["cu_ohm_m"] * BUS["wire_len_m"] / (BUS["wire_mm2"] * 1e-6)  # rho*L/A
BUS_PACK_AH = ELEC["batt_kwh"] * 1000.0 / BUS["pack_v_nom"]                 # pack capacity (Ah)

# ENGINE/TRACTION CONTROL UNIT (ECU). The control logic (engine on/off, regen split,
# traction current, SOC window, predictive grade) is a deterministic fixed-rate loop --
# a real microcontroller cadence -- so timing is honest and maintainable. Thresholds
# live in ENGINE_*_SOC / ENGINE_OFF_TEMP_C and PREDICT_* above; this sets the loop rate
# the sim advances that logic at, and the max rate the ECU may slew the rail current.
ECU = {
    "loop_hz":          200.0,   # control-loop update rate (5 ms period)
    "current_slew_a_s": 6000.0,  # max rail-current command slew (A/s)
}


# =============================================================================
# SECTION 2 -- COLORS & THEME
# =============================================================================

BG_TOP      = (14, 18, 26)
BG_BOT      = (4, 6, 10)
C_RING      = (96, 110, 128)
C_RING_HOT  = (255, 150, 40)
C_CHAMBER   = (70, 78, 92)
C_CHAMBER_FIRE = (255, 110, 30)
C_FLYWHEEL  = (150, 60, 55)      # tungsten-composite (warm metal)
C_FLYWHEEL_BAR = (220, 120, 110)
C_CLUTCH    = (60, 140, 165)
C_SHAFT     = (200, 205, 215)
C_TRANS     = (110, 118, 135)
C_TRANS_ALT = (78, 86, 104)
C_GEN       = (70, 95, 150)
C_OUTGEAR   = (210, 180, 70)
C_TEXT      = (224, 230, 238)
C_TEXT_DIM  = (150, 160, 175)
C_ACCENT    = (90, 200, 255)
C_GOOD      = (90, 220, 130)
C_WARN      = (255, 200, 60)
C_BAD       = (255, 90, 90)
C_PANEL     = (18, 24, 34)
C_PANEL_HI  = (30, 40, 56)
C_ROAD      = (40, 44, 52)
C_ROAD_LINE = (210, 210, 120)
C_GRASS     = (26, 40, 30)
C_SKY1      = (60, 110, 170)
C_SKY2      = (150, 190, 220)


# =============================================================================
# SECTION 3 -- MINI 3D ENGINE (software renderer, painter's algorithm)
# =============================================================================

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


class Mesh:
    """A bag of vertices + polygon faces with a base color. Coordinates are in
    metres, engine axis along +Z. `spin` is the rotation ratio relative to the
    master engine angle; `pivot` offsets the local origin; `tilt` is a static
    (rx, ry) rotation used for the angled output gear."""

    def __init__(self, verts, faces, color, name="", spin=1.0, group="default",
                 pivot=(0.0, 0.0, 0.0), tilt=(0.0, 0.0), hot=False,
                 selectable=False):
        self.verts = np.asarray(verts, dtype=float)
        self.faces = faces
        self.color = color
        self.name = name
        self.spin = spin
        self.group = group
        self.pivot = np.asarray(pivot, dtype=float)
        self.tilt = tilt
        self.hot = hot            # block metal that reddens with engine temperature
        self.chamber_index = None
        self.selectable = selectable
        self.brake = False        # rear light bar: brightens red when braking
        self.reverse = False      # reverse light: brightens white in reverse

    def world_verts(self, angle, selector_radius=None):
        v = self.verts
        if self.spin:
            v = v @ rot_z(angle * self.spin).T        # spin about the part's OWN axis
        rx, ry = self.tilt
        if rx or ry:
            v = v @ (rot_x(rx) @ rot_y(ry)).T         # then orient it (tilt into place)
        pivot = self.pivot.copy()
        if self.selectable and selector_radius is not None:
            pivot[0] = selector_radius
        return v + pivot


# ---- primitive builders ----------------------------------------------------

def _detail_seg(seg):
    return max(8, int(round(seg * VISUAL_DETAIL)))


def _annulus_cylinder(r_out, r_in, z0, z1, seg=44):
    """Hollow tube (ring) closed at both axial ends. Returns (verts, faces)."""
    seg = _detail_seg(seg)
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r_out * math.cos(a), r_out * math.sin(a), z))
        for a in ang:
            verts.append((r_in * math.cos(a), r_in * math.sin(a), z))

    def oo(layer, i):
        return layer * (2 * seg) + (i % seg)

    def ii(layer, i):
        return layer * (2 * seg) + seg + (i % seg)

    for i in range(seg):
        faces.append((oo(0, i), oo(0, i + 1), oo(1, i + 1), oo(1, i)))   # outer wall
        faces.append((ii(0, i), ii(1, i), ii(1, i + 1), ii(0, i + 1)))   # inner wall
        faces.append((oo(0, i), ii(0, i), ii(0, i + 1), oo(0, i + 1)))   # front cap
        faces.append((oo(1, i), oo(1, i + 1), ii(1, i + 1), ii(1, i)))   # back cap
    return verts, faces


def _solid_cylinder(r, z0, z1, seg=40):
    seg = _detail_seg(seg)
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r * math.cos(a), r * math.sin(a), z))
    c0 = len(verts)
    verts.append((0, 0, z0))
    c1 = len(verts)
    verts.append((0, 0, z1))
    for i in range(seg):
        a, b = i, (i + 1) % seg
        faces.append((a, b, seg + b, seg + a))
        faces.append((c0, b, a))
        faces.append((c1, seg + a, seg + b))
    return verts, faces


def _box(cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [(cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
         (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
         (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
         (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz)]
    f = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
         (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return v, f


def _hull(sections):
    """Loft a smooth body shell from a list of rectangular cross-sections along Z.
    Each section is (z, halfwidth, y_bottom, y_top): a rectangle in the X-Y plane at
    that Z. Consecutive sections are skinned with quads and the ends are capped.
    Used to build the whole-car teardrop body / canopy to scale."""
    verts, faces = [], []
    rings = []
    for (z, hw, y0, y1) in sections:
        base = len(verts)
        verts += [(-hw, y0, z), (hw, y0, z), (hw, y1, z), (-hw, y1, z)]
        rings.append(base)
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((a + k, a + k2, b + k2, b + k))
    faces.append((rings[0], rings[0] + 1, rings[0] + 2, rings[0] + 3))       # front cap
    last = rings[-1]
    faces.append((last + 3, last + 2, last + 1, last))                        # rear cap
    return verts, faces


def _smooth_sections(ctrl, sub=3):
    """Catmull-Rom subdivide a list of loft cross-sections (each a 4-tuple such as
    (z, halfwidth, y_bottom, y_top)) into `sub`x as many, for a smoother skinned
    shell that still passes through the original control sections."""
    pts = [np.array(s, dtype=float) for s in ctrl]
    n = len(pts)
    out = []
    for i in range(n - 1):
        p0, p1, p2, p3 = pts[max(0, i - 1)], pts[i], pts[i + 1], pts[min(n - 1, i + 2)]
        for j in range(sub):
            t = j / sub
            t2, t3 = t * t, t * t * t
            s = 0.5 * ((2 * p1) + (-p0 + p2) * t
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(tuple(s))
    out.append(tuple(pts[-1]))
    return out


def _arc_shell(r_out, r_in, z0, z1, a0, a1, seg=18):
    """Partial annulus (OPEN arc) shell between angles a0..a1: outer + inner walls,
    two axial caps and two radial end caps. Axis along local Z (rotate/position it
    after). Used for wheel-arch fenders over the top of each tyre."""
    seg = _detail_seg(seg)
    verts, faces = [], []
    ang = np.linspace(a0, a1, seg + 1)
    n = seg + 1
    for z in (z0, z1):
        for a in ang:
            verts.append((r_out * math.cos(a), r_out * math.sin(a), z))
        for a in ang:
            verts.append((r_in * math.cos(a), r_in * math.sin(a), z))

    def oo(layer, i):
        return layer * (2 * n) + i

    def ii(layer, i):
        return layer * (2 * n) + n + i

    for i in range(seg):
        faces.append((oo(0, i), oo(0, i + 1), oo(1, i + 1), oo(1, i)))     # outer wall
        faces.append((ii(0, i), ii(1, i), ii(1, i + 1), ii(0, i + 1)))     # inner wall
        faces.append((oo(0, i), ii(0, i), ii(0, i + 1), oo(0, i + 1)))     # z0 cap
        faces.append((oo(1, i), oo(1, i + 1), ii(1, i + 1), ii(1, i)))     # z1 cap
    faces.append((oo(0, 0), oo(1, 0), ii(1, 0), ii(0, 0)))                 # a0 end cap
    faces.append((oo(0, seg), ii(0, seg), ii(1, seg), oo(1, seg)))         # a1 end cap
    return verts, faces


class Part:
    """A named, spec'd logical component made of one or more meshes. Carries an
    assembly `order`, an `explode` offset (where it slides to when disassembled)
    and a `specs` list shown in the inspector card."""

    def __init__(self, key, name, meshes, specs, order, explode, color):
        self.key = key
        self.name = name
        self.meshes = meshes
        self.specs = specs
        self.order = order
        self.explode = np.asarray(explode, dtype=float)
        self.color = color
        n = np.linalg.norm(self.explode)
        self.popdir = self.explode / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def _grp(meshes, group):
    """Tag every mesh with the kinematic group it spins with; returns the list."""
    for m in meshes:
        m.group = group
    return meshes


def _place_spinner(meshes, pivot, tilt, group):
    """Turn a set of origin-built meshes into an OFF-AXIS spinner: they spin about
    their own local Z, get tilted into place, then pivoted to `pivot`. Used for
    the steam turbine, 2nd-stage turbine and the regen road wheel."""
    piv = np.asarray(pivot, dtype=float)
    for m in meshes:
        m.pivot = piv.copy()
        m.tilt = tilt
        m.group = group
    return meshes


def _differential(zg, rg, rog):
    """Output adapter: a differential housing + crown gear + two axle stubs that
    complete the driveline to the wheels (the 'wheel spin output' side view)."""
    meshes = []
    cx = rg + rog + 0.16
    v, f = _box(cx, 0, zg, 0.15, 0.18, 0.18)
    meshes.append(Mesh(v, f, (70, 150, 110), group="static"))
    v, f = _annulus_cylinder(0.085, 0.05, zg - 0.022, zg + 0.022, seg=32)
    v = np.asarray(v) + np.array([cx - 0.06, 0, 0])
    meshes.append(Mesh(v, f, (150, 150, 90), group="diff"))
    half = 0.16
    for sgn in (1, -1):
        v, f = _solid_cylinder(0.03, -half, half, seg=18)
        v = np.asarray(v) @ rot_y(math.pi / 2).T
        v = v + np.array([cx + sgn * (0.075 + half), 0, zg])
        meshes.append(Mesh(v, f, (60, 64, 72), group="diff"))
        v, f = _solid_cylinder(0.07, -0.02, 0.02, seg=22)
        v = np.asarray(v) @ rot_y(math.pi / 2).T
        v = v + np.array([cx + sgn * (0.075 + 2 * half), 0, zg])
        meshes.append(Mesh(v, f, (30, 32, 36), group="diff"))
    return meshes


def _combustion_chamber(a, zc, chamber_index):
    """A single, to-scale CONCAVE combustion pocket (a water-wheel 'bucket').

    Built in a local frame (X = radial, opening outward; Y = tangential width;
    Z = axial), then rotated to angle `a` and translated to the pocket centre.
    The floor is a genuine concave scoop (deepest at the middle) so the sealed
    explosion is caught and driven into rotation instead of leaking around the
    rim. Dimensions come straight from DIMS, so the geometry is to scale.
    Returns (pocket_mesh, [detail_meshes])."""
    mm = 0.001
    r_out = DIMS["ring_outer_d_mm"] * mm / 2
    cw = DIMS["chamber_width_mm"] * mm        # tangential width
    cd = DIMS["chamber_depth_mm"] * mm        # radial depth
    th = DIMS["ring_thickness_mm"] * mm
    hz = th * 0.40                             # axial half-height of the pocket
    rr = r_out - cd / 2                        # pocket radial centre
    nseg = 6

    verts, faces = [], []

    def add(p):
        verts.append(p)
        return len(verts) - 1

    fi, oi = [], []                            # floor / opening vertex indices
    for j in range(nseg + 1):
        t = -cw / 2 + cw * j / nseg
        # concave floor: deepest (most negative X) at the centre, shallow at the
        # tangential edges -> a scoop that catches the combustion load.
        fx = -cd / 2 + 0.30 * cd * (2 * t / cw) ** 2
        fi.append((add((fx, t, -hz)), add((fx, t, hz))))
        oi.append((add((cd / 2, t, -hz)), add((cd / 2, t, hz))))
    for j in range(nseg):
        # concave floor wall (the working face of the bucket)
        faces.append((fi[j][0], fi[j][1], fi[j + 1][1], fi[j + 1][0]))
        # -Z and +Z side caps of the pocket
        faces.append((fi[j][0], oi[j][0], oi[j + 1][0], fi[j + 1][0]))
        faces.append((fi[j][1], fi[j + 1][1], oi[j + 1][1], oi[j][1]))
    # tangential end walls (leading / trailing) -- opening left open to look in
    faces.append((fi[0][0], oi[0][0], oi[0][1], fi[0][1]))
    faces.append((fi[nseg][0], fi[nseg][1], oi[nseg][1], oi[nseg][0]))

    v = np.asarray(verts, float) @ rot_z(a).T + np.array(
        [rr * math.cos(a), rr * math.sin(a), zc])
    pocket = Mesh(v, faces, C_CHAMBER, group="piston", hot=True)
    pocket.chamber_index = chamber_index

    # leading + trailing sealing apex ridges (the metal tips between buckets)
    detail = []
    for sgn in (-1.0, 1.0):
        ax, ay = (r_out * 0.99) * math.cos(a), (r_out * 0.99) * math.sin(a)
        bx = ax - sgn * (cw * 0.5) * math.sin(a)
        by = ay + sgn * (cw * 0.5) * math.cos(a)
        vb, fb = _box(bx, by, zc, 0.012, 0.016, th * 0.86)
        vb = (np.asarray(vb) - np.array([bx, by, zc])) @ rot_z(a).T + np.array([bx, by, zc])
        detail.append(Mesh(vb, fb, (150, 160, 175), group="piston", hot=True))
    return pocket, detail


def _rotary_ring(zc):
    """One rotary combustion piston-ring + its 8 concave combustion chambers at
    axial zc. Chamber pocket meshes carry chamber_index 0..7 (shared firing
    schedule). The pockets are detailed, to-scale concave scoops."""
    mm = 0.001
    r_out = DIMS["ring_outer_d_mm"] * mm / 2
    r_in = DIMS["ring_inner_d_mm"] * mm / 2
    th = DIMS["ring_thickness_mm"] * mm
    v, f = _annulus_cylinder(r_out, r_in, zc - th / 2, zc + th / 2, seg=60)
    meshes = [Mesh(v, f, C_RING, group="piston", hot=True)]
    for i in range(DIMS["chambers"]):
        a = i * 2 * math.pi / DIMS["chambers"]
        pocket, detail = _combustion_chamber(a, zc, i)
        meshes.append(pocket)
        meshes.extend(detail)
    # bored cooling tunnels through the block (coolant-blue, rotate with block)
    rt = (r_out + r_in) / 2
    for k in range(DIMS["chambers"]):
        a = (k + 0.5) * 2 * math.pi / DIMS["chambers"]
        x, y = rt * math.cos(a), rt * math.sin(a)
        v, f = _solid_cylinder(0.016, zc - th / 2 - 0.004, zc + th / 2 + 0.004, seg=6)
        v = np.asarray(v) + np.array([x, y, 0.0])
        meshes.append(Mesh(v, f, (55, 120, 190), group="piston"))
    return meshes


def _injector(zc):
    """A fixed (non-spinning) direct-injection nozzle at the top of the rim."""
    mm = 0.001
    r_out = DIMS["ring_outer_d_mm"] * mm / 2
    v, f = _box(0.0, r_out * 1.06, zc, 0.05, 0.11, 0.05)
    nozzle = Mesh(v, f, (230, 200, 90), group="static")
    v, f = _box(0.0, r_out * 0.97, zc, 0.018, 0.06, 0.018)
    tip = Mesh(v, f, (255, 230, 120), group="static")
    return [nozzle, tip]


def _compression_gate(zc):
    """Pressure seal and charge assist for the circular chamber problem.
    A stationary sealed shoe closes over the chamber near the injector while
    turbo/supercharger boost and optional internal air-PSI jets raise charge
    density inside the chamber."""
    mm = 0.001
    r_out = DIMS["ring_outer_d_mm"] * mm / 2
    th = DIMS["ring_thickness_mm"] * mm
    meshes = []
    # Sealed arc shoe hugging the top of the rotary ring.
    for deg in (62, 74, 86, 98, 110):
        a = math.radians(deg)
        cx, cy = (r_out + 0.028) * math.cos(a), (r_out + 0.028) * math.sin(a)
        v, f = _box(cx, cy, zc, 0.070, 0.020, th * 0.88)
        v = (np.asarray(v) - np.array([cx, cy, zc])) @ rot_z(a).T + np.array([cx, cy, zc])
        meshes.append(Mesh(v, f, (80, 155, 135), group="static"))
    # Intake reed port before the compression zone.
    a = math.radians(56)
    px, py = (r_out + 0.060) * math.cos(a), (r_out + 0.060) * math.sin(a)
    v, f = _box(px, py, zc, 0.060, 0.034, th * 0.45)
    v = (np.asarray(v) - np.array([px, py, zc])) @ rot_z(a).T + np.array([px, py, zc])
    meshes.append(Mesh(v, f, (75, 150, 205), group="static"))
    # Optional internal PSI jets live inside the sealed chamber pocket only.
    for deg in (82, 90, 98):
        a = math.radians(deg)
        jx, jy = (r_out - 0.050) * math.cos(a), (r_out - 0.050) * math.sin(a)
        v, f = _solid_cylinder(0.007, zc - th * 0.28, zc - th * 0.12, seg=8)
        v = np.asarray(v) + np.array([jx, jy, 0.0])
        meshes.append(Mesh(v, f, (90, 210, 235), group="static"))
    # Radial compression ram at the injector zone, cam driven from the core.
    meshes.append(_pipe((0.0, r_out + 0.040, zc),
                        (0.0, r_out + 0.155, zc), 0.020, (105, 180, 155)))
    v, f = _solid_cylinder(0.040, -0.020, 0.020, seg=18)
    meshes.append(Mesh(v, f, (175, 190, 125), group="core",
                       pivot=(0.0, r_out + 0.195, zc)))
    v, f = _box(0.020, 0.018, 0.0, 0.060, 0.030, 0.030)
    meshes.append(Mesh(v, f, (190, 205, 135), group="core",
                       pivot=(0.0, r_out + 0.195, zc)))
    return meshes


def _clutch_stack(radius, zc, total_th, grp_in, grp_out, plates=5):
    """A visible wet multi-plate clutch: friction plates (output side) interleaved
    with steel separator plates (input side). When slipping, the two groups spin
    at different rates so the slip is visible. Includes a hub + lock bolts."""
    meshes = []
    pthk = total_th / plates
    for i in range(plates):
        z0 = zc - total_th / 2 + i * pthk
        out = (i % 2 == 0)
        r = radius * (1.0 if out else 0.92)
        col = (95, 165, 190) if out else (150, 150, 160)
        v, f = _solid_cylinder(r, z0 + pthk * 0.12, z0 + pthk * 0.82, seg=24)
        meshes.append(Mesh(v, f, col, group=(grp_out if out else grp_in)))
    v, f = _solid_cylinder(radius * 0.34, zc - total_th / 2, zc + total_th / 2, seg=14)
    meshes.append(Mesh(v, f, (70, 74, 82), group=grp_out))
    meshes += _grp(_bolt_ring(radius * 0.86, zc, 6, 0.009, total_th * 0.55,
                              (200, 210, 220)), grp_out)
    return meshes


def _bolt_ring(radius, zc, count, bolt_r, half_len, col):
    """A circle of small bolt-head cylinders (detail) about the engine axis."""
    meshes = []
    for k in range(count):
        a = k * 2 * math.pi / count
        v, f = _solid_cylinder(bolt_r, zc - half_len, zc + half_len, seg=6)
        v = np.asarray(v) + np.array([radius * math.cos(a), radius * math.sin(a), 0.0])
        meshes.append(Mesh(v, f, col))
    return meshes


def _fin_ring(radius, zc, count, col, fl=0.05, fw=0.018, fz=0.07):
    """A ring of radial fins / stator slots (detail) standing off a rim."""
    meshes = []
    for k in range(count):
        a = k * 2 * math.pi / count
        cx, cy = radius * math.cos(a), radius * math.sin(a)
        v, f = _box(cx, cy, zc, fl, fw, fz)
        v = (np.asarray(v) - np.array([cx, cy, zc])) @ rot_z(a).T + np.array([cx, cy, zc])
        meshes.append(Mesh(v, f, col))
    return meshes


def _gear_teeth(radius, zc, count, half_th, col, pivot, tilt, selectable=False):
    """A ring of gear teeth in a part's LOCAL frame (pivot/tilt applied later)."""
    meshes = []
    for k in range(count):
        a = k * 2 * math.pi / count
        cx, cy = radius * math.cos(a), radius * math.sin(a)
        v, f = _box(cx, cy, zc, 0.016, 0.013, half_th * 2)
        v = (np.asarray(v) - np.array([cx, cy, zc])) @ rot_z(a).T + np.array([cx, cy, zc])
        meshes.append(Mesh(v, f, col, pivot=pivot, tilt=tilt, selectable=selectable))
    return meshes


def trans_ratios():
    edges = trans_ring_edges()
    pitch_radii = [(inner + outer) * 0.5 for inner, outer in edges]
    base = max(1e-6, pitch_radii[0])
    return [r / base for r in pitch_radii]


def combustion_count(value):
    """Firing rate is selectable from 1 up to all 8 combustions per single
    revolution (idle/cruise = 1, full power = 8, and every step in between)."""
    return max(1, min(DIMS["chambers"], int(round(value))))


def combustion_duty(value):
    return combustion_count(value) / float(DIMS["chambers"])


def boost_multiplier(supercharger_engage):
    return 1.0 + SUPERCHARGER_BOOST_GAIN * max(0.0, min(1.0, supercharger_engage))


def trans_ring_edges():
    """Return (inner, outer) radii for each transmission ring, centre outward.
    Gear 1 is the center ring; gear N is the outermost ring. Each ring carries
    teeth at its inner edge so the movable receival gear can shift outward."""
    mm = 0.001
    nrings = DIMS["trans_rings"]
    ro = DIMS["trans_outer_d_mm"] * mm / 2
    ri = DIMS["trans_inner_d_mm"] * mm / 2
    step = (ro - ri) / nrings
    return [(ri + step * k, ri + step * (k + 1)) for k in range(nrings)]


def output_gear_mesh_radius(gear):
    """Center radius for the small generator receival gear at selected ring."""
    edges = trans_ring_edges()
    gear = max(0, min(len(edges) - 1, int(gear)))
    inner, outer = edges[gear]
    rog = DIMS["out_gear_d_mm"] * 0.001 / 2
    # Small side-mounted pinion meshes at the selected ring's inner-edge teeth.
    return min(outer - rog * 0.35, inner + rog + 0.005)


def _integral_clutch(zc, plates=6):
    """The MAIN wet clutch, integral to the piston and sitting in its inner bore.

    It is a MULTI-PLATE wet clutch (not a single disc): steel separator plates
    splined to the piston's clutch DRUM interleave with friction plates splined
    to the central Core hub. More plates = more slip surface, so the clutch can
    modulate a lot of controlled slip before it locks 1:1. The piston gives the
    clutch a real 'spot' to grip: a clutch drum/basket built into its inner bore.
    When slipping, the piston-keyed plates and core-keyed plates spin at
    different rates so the slip is visible."""
    mm = 0.001
    r_in = DIMS["ring_inner_d_mm"] * mm / 2
    r_drum = r_in * 0.92
    stack_th = DIMS["ring_thickness_mm"] * mm * 0.60
    meshes = []
    # Clutch basket / engagement DRUM built into the piston inner bore -- this is
    # the spot the main wet clutch grips onto (splined to the piston wall).
    v, f = _annulus_cylinder(r_drum * 1.07, r_drum * 0.99,
                             zc - stack_th / 2 - 0.010, zc + stack_th / 2 + 0.010, seg=36)
    meshes.append(Mesh(v, f, (70, 120, 140), group="piston"))
    meshes += _grp(_fin_ring(r_drum, zc, 8, (95, 155, 175),
                             fl=0.012, fw=0.012, fz=stack_th * 0.92), "piston")
    # Core hub the friction plates key onto (splined to the central Core).
    v, f = _solid_cylinder(r_in * 0.34, zc - stack_th / 2 - 0.012, zc + stack_th / 2 + 0.012, seg=20)
    meshes.append(Mesh(v, f, (150, 150, 160), group="core"))
    meshes += _grp(_fin_ring(r_in * 0.37, zc, 8, (175, 175, 185),
                             fl=0.012, fw=0.010, fz=stack_th * 0.92), "core")
    # Interleaved MULTI-PLATE stack: steel separators (piston-keyed) alternate
    # with friction plates (core-keyed). More plates = more slip surface.
    pthk = stack_th / plates
    for i in range(plates):
        z0 = zc - stack_th / 2 + i * pthk
        piston_plate = (i % 2 == 0)
        r = r_drum * 0.96 if piston_plate else r_drum * 0.72
        col = (150, 150, 160) if piston_plate else (95, 165, 190)
        grp = "piston" if piston_plate else "core"
        v, f = _solid_cylinder(r, z0 + pthk * 0.14, z0 + pthk * 0.80, seg=26)
        meshes.append(Mesh(v, f, col, group=grp))
    return meshes


def _pipe(p0, p1, r, col, seg=8):
    """A straight pipe (cylinder) between two 3D points."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d)) or 1e-6
    v, f = _solid_cylinder(r, 0.0, L, seg=seg)
    zaxis = d / L
    tmp = np.array([0.0, 0.0, 1.0]) if abs(zaxis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    xaxis = np.cross(tmp, zaxis)
    xaxis /= (np.linalg.norm(xaxis) or 1e-6)
    yaxis = np.cross(zaxis, xaxis)
    R = np.stack([xaxis, yaxis, zaxis], axis=1)
    v = np.asarray(v) @ R.T + p0
    return Mesh(v, f, col, group="static")


def _cooling_ring(zc, r_out):
    """The COOLING RING around the piston: a slim closed-loop jacket hugging the
    rim that ABSORBS the block's combustion + friction heat into the working fluid
    (the point is to keep heat, not shed it). A collector riser carries the hot
    fluid down to the compact boiler. Kept thin and to scale."""
    mm = 0.001
    gap = DIMS["coolring_gap_mm"] * mm
    jt = DIMS["coolring_thick_mm"] * mm
    th = DIMS["ring_thickness_mm"] * mm
    ri = r_out + gap
    ro = ri + jt
    meshes = []
    # slim cooling-ring jacket hugging the piston OD (hot -> reddens with temp)
    v, f = _annulus_cylinder(ro, ri, zc - th * 0.85, zc + th * 0.85, seg=54)
    meshes.append(Mesh(v, f, (150, 92, 70), group="static", hot=True))
    # short pick-up fingers that touch the rim (absorb heat into the fluid)
    for kk in range(12):
        a = kk * 2 * math.pi / 12
        cx, cy = (r_out + gap * 0.5) * math.cos(a), (r_out + gap * 0.5) * math.sin(a)
        v, f = _box(cx, cy, zc, gap, 0.010, th * 0.6)
        v = (np.asarray(v) - np.array([cx, cy, zc])) @ rot_z(a).T + np.array([cx, cy, zc])
        meshes.append(Mesh(v, f, (124, 82, 66), group="static", hot=True))
    # hot-fluid collector riser routed down toward the boiler focus
    meshes.append(_pipe((0.0, -ri, zc), (0.0, -(r_out + 0.16), zc), 0.010, (200, 120, 90)))
    return meshes


def _heat_shield(zc, r_out):
    """GEOMETRIC HEAT SHIELD (replaces the old heat-absorbing metal ingots): angled
    reflector vanes form a funnel that DIRECTS the trapped heat onto one focus area
    of the boiler, instead of a mass soaking it up. Geometry aims the heat."""
    mm = 0.001
    span = DIMS["shield_span_mm"] * mm
    n = DIMS["shield_vanes"]
    meshes = []
    fx, fy = 0.0, -(r_out + 0.20)          # focus point (top of the boiler)
    # curved parabola-ish funnel of angled vanes converging on the focus
    for i in range(n):
        t = (i / (n - 1)) - 0.5            # -0.5 .. 0.5 across the funnel
        vx = t * span                      # vane lateral position
        vy = fy - 0.10 - abs(t) * 0.14     # deeper at the edges -> concave dish
        # each vane tilts to aim its face at the focus
        aim = math.atan2(fx - vx, (fy - vy) or 1e-6)
        cxp, cyp = vx, vy
        v, f = _box(cxp, cyp, zc, 0.012, 0.085, 0.11)
        v = (np.asarray(v) - np.array([cxp, cyp, zc])) @ rot_z(aim).T \
            + np.array([cxp, cyp, zc])
        meshes.append(Mesh(v, f, (170, 176, 186), group="static"))   # reflective
    # backing dish tying the vanes together
    v, f = _box(fx, fy - 0.20, zc, span * 1.02, 0.02, 0.13)
    meshes.append(Mesh(v, f, (95, 100, 110), group="static"))
    # a couple of directed-heat rays converging on the focus (visual aim lines)
    for t in (-0.32, 0.0, 0.32):
        meshes.append(_pipe((t * span, fy - 0.16, zc), (fx, fy + 0.02, zc),
                            0.004, (210, 150, 90)))
    return meshes


def _steam_boiler(zc, r_out):
    """COMPACT closed-loop steam pack (to scale): a small boiler drum fed by the
    directed heat, a DUAL-CHAMBER alternating valve block (two chambers so there is
    no dead stroke), a MULTI-STAGE compound expander (pressure reused stage to
    stage until weak), a compact GENERATOR, a pressure-RELEASE vent and a condenser
    that closes the loop. Sits just below the engine (-Y)."""
    mm = 0.001
    meshes = []
    by = -(r_out + 0.24)                   # compact: tucked close under the engine
    bd = DIMS["boiler_d_mm"] * mm / 2
    bl = DIMS["boiler_len_mm"] * mm / 2
    # boiler drum (horizontal), HOT (kept hot by directed heat + fluid inertia)
    v, f = _solid_cylinder(bd, -bl, bl, seg=24)
    v = np.asarray(v) @ rot_y(math.pi / 2).T + np.array([0.0, by, zc])
    meshes.append(Mesh(v, f, (176, 96, 62), group="static", hot=True))
    v, f = _annulus_cylinder(bd * 1.05, bd * 0.98, -bl * 0.5, bl * 0.5, seg=24)
    v = np.asarray(v) @ rot_y(math.pi / 2).T + np.array([0.0, by, zc])
    meshes.append(Mesh(v, f, (66, 128, 182), group="static"))
    # DUAL alternating valve/expansion chambers (the two-chamber switch system)
    scd = DIMS["steam_chamber_d_mm"] * mm / 2
    for sgn in (-1.0, 1.0):
        cxp = sgn * (bd * 0.55)
        v, f = _solid_cylinder(scd, -0.02, 0.02, seg=16)
        v = np.asarray(v) @ rot_y(math.pi / 2).T + np.array([cxp, by - bd - 0.03, zc])
        meshes.append(Mesh(v, f, (120, 150, 175), group="static"))
        v, f = _box(cxp, by - bd - 0.03, zc, 0.03, 0.05, 0.05)
        meshes.append(Mesh(v, f, (90, 120, 150), group="static"))
    # MULTI-STAGE compound expander: 3 stage wheels of growing size on one axis
    ssd = DIMS["steam_stage_d_mm"] * mm / 2
    ax0 = bl + 0.05
    for s in range(3):
        piv = np.array([ax0 + s * 0.055, by, zc])
        r = ssd * (0.70 + 0.18 * s)        # each stage larger (pressure expands)
        hub = _solid_cylinder(r * 0.34, -0.016, 0.016, seg=14)
        meshes += _place_spinner([Mesh(hub[0], hub[1], (150, 160, 175))],
                                 piv, (0.0, math.pi / 2), "steam")
        meshes += _place_spinner(_fin_ring(r * 0.92, 0.0, 12, (188, 198, 212),
                                           fl=0.024, fw=0.006, fz=0.03),
                                 piv, (0.0, math.pi / 2), "steam")
    # compact steam GENERATOR at the end of the compound axis
    sgd = DIMS["steam_gen_d_mm"] * mm / 2
    sgt = DIMS["steam_gen_thick_mm"] * mm
    gpiv = np.array([ax0 + 3 * 0.055 + sgt * 0.5, by, zc])
    gen = _solid_cylinder(sgd, -sgt / 2, sgt / 2, seg=20)
    meshes += _place_spinner([Mesh(gen[0], gen[1], (70, 150, 210))],
                             gpiv, (0.0, math.pi / 2), "steam")
    meshes += _place_spinner(_fin_ring(sgd * 0.95, 0.0, 14, (58, 90, 150),
                                       fl=0.024, fw=0.008, fz=sgt * 0.6),
                             gpiv, (0.0, math.pi / 2), "steam")
    # pressure-RELEASE vent turbine on top of the drum (regulates by venting PSI)
    rd = DIMS["relief_d_mm"] * mm / 2
    rpiv = np.array([0.0, by + bd + 0.04, zc])
    rh = _solid_cylinder(rd * 0.5, -0.015, 0.015, seg=12)
    meshes += _place_spinner([Mesh(rh[0], rh[1], (150, 160, 175))], rpiv, (0.0, 0.0), "steam")
    meshes += _place_spinner(_fin_ring(rd, 0.0, 8, (205, 210, 220),
                                       fl=0.014, fw=0.005, fz=0.015), rpiv, (0.0, 0.0), "steam")
    # condenser (closes the loop) at -X
    v, f = _box(-(bl + 0.08), by, zc, 0.07, 0.10, 0.07)
    meshes.append(Mesh(v, f, (50, 110, 165), group="static"))
    # loop pipes: cooling-ring riser -> drum, drum -> expander, condenser return
    meshes.append(_pipe((0.0, by + bd, zc), (0.0, -(r_out + 0.14), zc), 0.010, (200, 120, 90)))
    meshes.append(_pipe((bl, by, zc), (ax0 - 0.01, by, zc), 0.010, (210, 216, 226)))
    meshes.append(_pipe((-bl, by, zc), (-(bl + 0.05), by, zc), 0.009, (70, 150, 210)))
    return meshes


def _second_stage_turbine(zc, r_out):
    """The 2nd-stage exhaust RECOVERY turbine: it sits downstream of the main
    turbo and catches more of the leftover exhaust (a small generator), then
    routes the remaining heat into the boiler jacket. Mounted at the lower rim."""
    mm = 0.001
    meshes = []
    a = math.radians(238.0)
    ndir = np.array([math.cos(a), math.sin(a), 0.0])
    base = ndir * (r_out + 0.19) + np.array([0.0, 0.0, zc + 0.02])
    td = DIMS["recov2_turbine_d_mm"] * mm / 2
    # turbine housing (hot leftover exhaust)
    v, f = _annulus_cylinder(td, td * 0.42, -0.032, 0.032, seg=22)
    v = np.asarray(v) @ rot_y(math.pi / 2).T + base
    meshes.append(Mesh(v, f, (150, 85, 55), group="static", hot=True))
    # turbine wheel spins on leftover exhaust
    hub = _solid_cylinder(td * 0.34, -0.02, 0.02, seg=16)
    meshes += _place_spinner([Mesh(hub[0], hub[1], (160, 168, 182))],
                             base, (0.0, math.pi / 2), "recov2")
    meshes += _place_spinner(_fin_ring(td * 0.85, 0.0, 14, (150, 160, 175),
                                       fl=0.03, fw=0.008, fz=0.05),
                             base, (0.0, math.pi / 2), "recov2")
    # small recovery generator on the same axis (further out along the axis = +X)
    gd = DIMS["recov2_gen_d_mm"] * mm / 2
    gpiv = base + np.array([0.10, 0.0, 0.0])
    gen = _solid_cylinder(gd, -0.045, 0.045, seg=18)
    meshes += _place_spinner([Mesh(gen[0], gen[1], (70, 150, 210))],
                             gpiv, (0.0, math.pi / 2), "recov2")
    # duct from the main turbo exit (upper-left) into this 2nd turbine
    turbo_a = math.radians(210.0)
    tdir = np.array([math.cos(turbo_a), math.sin(turbo_a), 0.0])
    meshes.append(_pipe(tdir * (r_out + 0.14) + np.array([0.0, 0.0, zc]),
                        base + ndir * (-td), 0.012, (170, 92, 56)))
    # remaining-heat duct from this turbine down to the boiler jacket
    meshes.append(_pipe(base, (0.0, -(r_out + 0.16), zc), 0.012, (150, 80, 55)))
    return meshes


def _regen_wheel(zc, r_out):
    """A road wheel with the rail-ring used as a GENERATOR. On braking / downhill
    the same electromagnetic rail that directly turns the rim is back-driven by
    wheel motion and pours momentum back into the caps + battery."""
    mm = 0.001
    wd = DIMS["wheel_d_mm"] * mm / 2
    ww = DIMS["wheel_width_mm"] * mm
    wg = DIMS["wheel_gen_d_mm"] * mm / 2
    cx = r_out + 0.80
    cy = -(r_out + 0.06)
    piv = np.array([cx, cy, zc])
    meshes = []
    # tire
    tire = _annulus_cylinder(wd, wd * 0.72, -ww / 2, ww / 2, seg=30)
    meshes += _place_spinner([Mesh(tire[0], tire[1], (28, 30, 34))],
                             piv, (0.0, math.pi / 2), "wheel")
    # rim
    rim = _annulus_cylinder(wd * 0.72, wd * 0.34, -ww * 0.4, ww * 0.4, seg=24)
    meshes += _place_spinner([Mesh(rim[0], rim[1], (120, 128, 140))],
                             piv, (0.0, math.pi / 2), "wheel")
    # spokes
    meshes += _place_spinner(_fin_ring(wd * 0.52, 0.0, 6, (92, 98, 110),
                                       fl=wd * 0.6, fw=0.020, fz=ww * 0.35),
                             piv, (0.0, math.pi / 2), "wheel")
    # rail-ring GENERATOR disc (charges on brake / downhill)
    gen = _solid_cylinder(wg, -ww * 0.3, ww * 0.3, seg=22)
    meshes += _place_spinner([Mesh(gen[0], gen[1], (70, 95, 150))],
                             piv, (0.0, math.pi / 2), "wheel")
    meshes += _place_spinner(_fin_ring(wg * 0.95, 0.0, 16, (58, 78, 128),
                                       fl=0.03, fw=0.010, fz=ww * 0.3),
                             piv, (0.0, math.pi / 2), "wheel")
    # regen power cable back toward the battery/caps
    meshes.append(_pipe((cx - wd * 0.1, cy + wd * 0.2, zc),
                        (r_out + 0.08, cy + 0.05, zc), 0.010, (90, 200, 255)))
    return meshes


def _pedal_assist(zc, r_out):
    """POP-OUT PASSENGER PEDAL generators -- one per seat. Pushing the foot pedals
    spins a crank + a mini GENERATOR that trickle-charges the battery, like pedaling
    a bike. Positionable and pressable forever, adding cumulative charge and range.
    Shown as a row of pedal+generator units on the cabin side (+X)."""
    mm = 0.001
    cr = DIMS["pedal_crank_d_mm"] * mm / 2
    gr = DIMS["pedal_gen_d_mm"] * mm / 2
    n = DIMS["seats"]
    meshes = []
    x0 = r_out + 0.55
    span = 0.62
    for i in range(n):
        zi = zc + (i - (n - 1) / 2.0) * (span / max(1, n - 1))
        piv = np.array([x0, 0.10, zi])
        # crank flywheel + two pedal arms/blocks (spin on the 'pedal' group)
        hub = _solid_cylinder(cr * 0.28, -0.014, 0.014, seg=12)
        meshes += _place_spinner([Mesh(hub[0], hub[1], (150, 160, 175))],
                                 piv, (0.0, math.pi / 2), "pedal")
        for arm in (0.0, math.pi):
            ax, ay = cr * 0.8 * math.cos(arm), cr * 0.8 * math.sin(arm)
            pb = _box(ax, ay, 0.0, cr * 0.9, 0.014, 0.014)
            meshes += _place_spinner([Mesh(pb[0], pb[1], (200, 190, 90))],
                                     piv, (0.0, math.pi / 2), "pedal")
            pd = _box(cr * 0.95 * math.cos(arm), cr * 0.95 * math.sin(arm), 0.0,
                      0.05, 0.03, 0.018)
            meshes += _place_spinner([Mesh(pd[0], pd[1], (60, 64, 72))],
                                     piv, (0.0, math.pi / 2), "pedal")
        # mini trickle generator behind the crank
        gpiv = np.array([x0 + 0.10, 0.10, zi])
        gen = _solid_cylinder(gr, -0.03, 0.03, seg=16)
        meshes += _place_spinner([Mesh(gen[0], gen[1], (70, 150, 210))],
                                 gpiv, (0.0, math.pi / 2), "pedal")
        # pop-out mounting arm + trickle cable back to the pack
        meshes.append(_pipe((x0 - 0.02, 0.10, zi), (r_out + 0.10, 0.02, zi),
                            0.008, (90, 96, 110)))
        meshes.append(_pipe((x0 + 0.13, 0.10, zi), (r_out + 0.10, -0.05, zi),
                            0.006, (90, 200, 255)))
    return meshes


def _ambient_harvest(zc, r_out):
    """The MULTI-LAYER ambient harvest suite -- the compounding, fuel-free trickle:
    a quantum-dot SOLAR film over the upper surface, four linear ELECTROMAGNETIC
    regenerative suspension dampers, and a triboelectric (TENG) underbody strip.
    Shown around the model as vehicle-level systems."""
    meshes = []
    # SOLAR film: arched quantum-dot panel over the top (+Y), with a cell grid
    yy = r_out + 0.58
    hw, hd = r_out * 1.25, r_out * 0.95
    v, f = _box(0.0, yy, zc, hw * 2, 0.014, hd * 2)
    meshes.append(Mesh(v, f, (38, 46, 92), group="static"))
    for gx in range(-4, 5):
        x = gx * (hw * 2) / 9.0
        meshes.append(_pipe((x, yy + 0.009, zc - hd), (x, yy + 0.009, zc + hd),
                            0.004, (72, 92, 150)))
    for gz in (-0.55, 0.0, 0.55):
        z = zc + gz * hd * 2
        meshes.append(_pipe((-hw, yy + 0.009, z), (hw, yy + 0.009, z), 0.004, (72, 92, 150)))
    for sx in (-hw * 0.8, hw * 0.8):
        meshes.append(_pipe((sx, yy, zc), (sx * 0.4, r_out + 0.05, zc), 0.008, (92, 98, 112)))
    meshes.append(_pipe((0.0, yy, zc), (0.0, r_out + 0.05, zc), 0.008, (90, 200, 255)))
    # 4 linear ELECTROMAGNETIC regenerative suspension dampers (corners)
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            bx = sx * (r_out + 0.30)
            bz = zc + sz * (r_out * 0.72)
            base = np.array([bx, -(r_out + 0.16), bz])
            v, f = _solid_cylinder(0.045, -0.09, 0.09, seg=14)
            v = np.asarray(v) @ rot_x(math.pi / 2).T + base
            meshes.append(Mesh(v, f, (86, 108, 150), group="static"))
            v, f = _solid_cylinder(0.020, 0.06, 0.20, seg=10)   # magnet rod
            v = np.asarray(v) @ rot_x(math.pi / 2).T + base
            meshes.append(Mesh(v, f, (180, 185, 195), group="static"))
            meshes.append(_pipe(tuple(base), (sx * (r_out + 0.06), -(r_out + 0.06), zc),
                                0.005, (90, 200, 255)))
    # triboelectric (TENG) underbody strip (to -X so it clears the boiler at -Y)
    tx = -(r_out + 0.30)
    v, f = _box(tx, -(r_out * 0.2), zc, 0.10, 0.012, r_out * 1.3)
    meshes.append(Mesh(v, f, (120, 90, 150), group="static"))
    for gz in range(-3, 4):
        z = zc + gz * (r_out * 1.3) / 7.0
        meshes.append(_pipe((tx - 0.05, -(r_out * 0.2) + 0.008, z),
                            (tx + 0.05, -(r_out * 0.2) + 0.008, z), 0.003, (160, 120, 200)))
    return meshes


def _solar_roof(zc, r_out):
    """Dedicated high-efficiency PHOTOVOLTAIC ROOF panel -- an extra solar layer
    ABOVE the quantum-dot ambient film, feeding free electric straight to the pack.
    Shown as a large gridded PV slab over the top (+Y) with a power cable down."""
    meshes = []
    yy = r_out + 0.86
    hw, hd = r_out * 1.70, r_out * 1.30
    v, f = _box(0.0, yy, zc, hw * 2, 0.016, hd * 2)          # PV slab
    meshes.append(Mesh(v, f, (18, 34, 78), group="static"))
    for gx in range(-6, 7):                                   # PV cell grid (silver)
        x = gx * (hw * 2) / 13.0
        meshes.append(_pipe((x, yy + 0.010, zc - hd), (x, yy + 0.010, zc + hd),
                            0.003, (150, 160, 185)))
    for gz in range(-4, 5):
        z = zc + gz * (hd * 2) / 9.0
        meshes.append(_pipe((-hw, yy + 0.010, z), (hw, yy + 0.010, z), 0.003, (150, 160, 185)))
    v, f = _box(-hw * 0.30, yy + 0.012, zc, hw * 0.5, 0.004, hd * 1.6)   # glass sheen
    meshes.append(Mesh(v, f, (90, 120, 175), group="static"))
    for sx in (-hw * 0.8, hw * 0.8):                          # support struts
        meshes.append(_pipe((sx, yy, zc), (sx * 0.5, r_out + 0.05, zc), 0.010, (92, 98, 112)))
    meshes.append(_pipe((0.0, yy, zc + hd * 0.6), (0.06, r_out + 0.05, zc),
                        0.007, (90, 200, 255)))               # power cable to pack
    return meshes


def _windshield_concentrator(zc, r_out):
    """Auto-adjusting windshield Fresnel / MAGNIFYING LENS that tracks the sun and
    focuses concentrated sunlight onto a receiver on the boiler -- free solar HEAT
    into the closed-loop fluid. Shown as a raked lens up front, a bright focused
    beam to a small receiver drum, and a hot pipe carrying that heat to the boiler."""
    mm = 0.001
    meshes = []
    rl = DIMS["windshield_lens_d_mm"] * mm / 2
    rr = DIMS["lens_receiver_d_mm"] * mm / 2
    tilt = math.radians(52.0)                     # windshield rake
    R = rot_x(-tilt)
    lc = np.array([0.0, r_out + 0.32, zc + r_out + 0.30])     # lens centre (front, high)
    v, f = _solid_cylinder(rl, -0.006, 0.006, seg=40)          # lens disc
    v = np.asarray(v) @ R.T + lc
    meshes.append(Mesh(v, f, (150, 200, 225), group="static"))
    for gr in (0.35, 0.60, 0.82):                              # Fresnel groove rings
        v, f = _annulus_cylinder(rl * gr, rl * gr - 0.012, -0.008, 0.008, seg=36)
        v = np.asarray(v) @ R.T + lc
        meshes.append(Mesh(v, f, (185, 220, 240), group="static"))
    for sx in (-1.0, 1.0):                                     # auto-adjust gimbal arms
        top = lc + R @ np.array([sx * rl * 0.92, 0.0, 0.0])
        meshes.append(_pipe(tuple(top), (sx * r_out * 0.6, r_out + 0.06, zc + r_out * 0.5),
                            0.010, (90, 96, 110)))
    rcv = np.array([0.0, -(r_out + 0.05), zc + r_out + 0.10])  # receiver drum (front, low)
    v, f = _solid_cylinder(rr, -0.05, 0.05, seg=20)
    v = np.asarray(v) @ rot_x(math.pi / 2).T + rcv
    meshes.append(Mesh(v, f, (60, 64, 72), group="static"))
    # concentrated-sun BEAM: bright focus line from the lens down to the receiver
    meshes.append(_pipe(tuple(lc), tuple(rcv), 0.016, (255, 225, 90)))
    meshes.append(_pipe(tuple((lc + rcv) / 2), tuple(rcv), 0.009, (255, 240, 150)))
    # hot heat pipe from the receiver into the boiler focus (-Y)
    meshes.append(_pipe(tuple(rcv), (0.0, -(r_out + 0.12), zc), 0.010, (230, 120, 60)))
    return meshes


def _hydro_skin(zc, r_out):
    """SUPER-HYDROPHOBIC + plastron 'slip' body-coating sample: a body-panel swatch
    with beaded water droplets rolling off it. The wet-drag-cutting skin that makes
    rain bead and roll off (no clinging water film) and retains an air 'plastron'
    that cuts skin friction. Shown as a small tilted swatch off to the +X side so it
    does not occlude the engine."""
    meshes = []
    base = np.array([r_out + 0.40, r_out + 0.25, zc])
    R = rot_z(math.radians(-20))
    v, f = _box(0.0, 0.0, 0.0, 0.24, 0.010, 0.17)               # coated panel swatch
    v = np.asarray(v) @ R.T + base
    meshes.append(Mesh(v, f, (110, 140, 170), group="static"))
    # beaded droplets standing proud (the super-hydrophobic high contact angle)
    beads = [(-0.08, -0.05), (-0.02, 0.03), (0.05, -0.04),
             (0.09, 0.05), (-0.05, 0.07), (0.02, -0.07)]
    for i, (dx, dz) in enumerate(beads):
        d = base + R @ np.array([dx, 0.014, dz])
        vv, ff = _solid_cylinder(0.016 - 0.0016 * i, -0.007, 0.007, seg=10)
        vv = np.asarray(vv) @ rot_x(math.pi / 2).T + d
        meshes.append(Mesh(vv, ff, (150, 205, 235), group="static"))
    for dz in (-0.05, 0.05):                                    # droplets rolling off
        a = base + R @ np.array([0.10, 0.014, dz])
        b = base + R @ np.array([0.16, -0.02, dz])
        meshes.append(_pipe(tuple(a), tuple(b), 0.005, (150, 205, 235)))
    return meshes


def _drag_groove_skin(zc, r_out):
    """PASSIVE DRAG-AIR CAPTURE skin: a body panel of directional micro-grooves
    (aligned WITH the airflow) that funnel the remaining boundary-layer air through
    an internal duct into a tiny BLADELESS recovery turbine + generator. Shown off to
    the +X side; the turbine spins on the 'dragturb' group."""
    meshes = []
    bx, by = r_out + 0.42, -0.02
    v, f = _box(bx, by, zc, 0.020, 0.09, 0.34)                  # grooved skin panel
    meshes.append(Mesh(v, f, (96, 126, 152), group="static"))
    for gy in range(-3, 4):                                     # grooves aligned with airflow (Z)
        y = by + gy * 0.09 / 3.5
        meshes.append(_pipe((bx + 0.011, y, zc - 0.16), (bx + 0.011, y, zc + 0.12),
                            0.0028, (150, 175, 200)))
    tpiv = np.array([bx + 0.03, by - 0.02, zc - 0.24])          # turbine location
    meshes.append(_pipe((bx + 0.01, by, zc + 0.10), tuple(tpiv), 0.012, (70, 90, 110)))  # duct
    tv, tf = _solid_cylinder(0.034, -0.014, 0.014, seg=18)      # bladeless turbine disc
    meshes += _place_spinner([Mesh(tv, tf, (120, 175, 215))], tpiv, (0.0, 0.0), "dragturb")
    for k in range(3):                                          # bladeless-disc rings (spin)
        rr = 0.034 * (0.45 + 0.16 * k)
        rv, rf = _annulus_cylinder(rr, rr - 0.004, -0.016, 0.016, seg=14)
        meshes += _place_spinner([Mesh(rv, rf, (90, 150, 195))], tpiv, (0.0, 0.0), "dragturb")
    gv, gf = _solid_cylinder(0.024, -0.02, 0.02, seg=14)        # mini generator
    gv = np.asarray(gv) + (tpiv + np.array([0.0, 0.0, -0.06]))
    meshes.append(Mesh(gv, gf, (70, 150, 210), group="static"))
    meshes.append(_pipe(tuple(tpiv + np.array([0.0, 0.0, -0.08])), (r_out + 0.06, -0.06, zc),
                        0.005, (90, 200, 255)))                 # cable to the pack
    return meshes


def _windmill_fin(zc, r_out):
    """PARKED WINDMILL / deployable rear tail fin: a vertical mast + stabiliser fin
    with a small VERTICAL-AXIS wind turbine on top that trickle-charges from ambient
    wind when parked. Blades spin on the 'windmill' group (about the vertical axis)."""
    meshes = []
    base = np.array([0.0, r_out + 0.10, zc - r_out - 0.22])
    mv, mf = _solid_cylinder(0.020, 0.0, 0.42, seg=12)          # vertical mast (stood up along +Y)
    mv = np.asarray(mv) @ rot_x(-math.pi / 2).T + base
    meshes.append(Mesh(mv, mf, (120, 130, 145), group="static"))
    fv, ff = _box(0.0, 0.22, 0.0, 0.012, 0.34, 0.16)            # stabiliser fin plate
    fv = np.asarray(fv) + base
    meshes.append(Mesh(fv, ff, (90, 110, 140), group="static"))
    hub = base + np.array([0.0, 0.44, 0.0])                     # turbine hub on top
    for k in range(3):                                          # 3 vertical-axis blades
        a = k * 2 * math.pi / 3
        bv, bf = _box(0.10 * math.cos(a), 0.10 * math.sin(a), 0.0, 0.05, 0.05, 0.18)
        meshes += _place_spinner([Mesh(bv, bf, (150, 205, 235))], tuple(hub),
                                 (math.pi / 2, 0.0), "windmill")
    gv, gf = _solid_cylinder(0.030, -0.03, 0.03, seg=14)        # hub generator
    gv = np.asarray(gv) @ rot_x(math.pi / 2).T + hub
    meshes.append(Mesh(gv, gf, (70, 150, 210), group="static"))
    meshes.append(_pipe(tuple(base), (r_out + 0.05, 0.0, zc), 0.005, (90, 200, 255)))  # cable
    return meshes


def _flow_through_duct(zc, r_out):
    """THROUGH-BODY STRAIGHT FLOW-THROUGH DUCT: a wide, thin, hollow underbody duct
    running front-to-rear so nose air flows 100% straight through the car and fills
    the rear wake (pressure-drag kill). Built as four thin walls (an OPEN bore, so it
    does not occlude the engine), a slotted front grill, a widening rear diffuser and
    bright straight-through flow lines. Scaled to the preview so it reads clearly."""
    mm = 0.001
    meshes = []
    w = DIMS["flow_duct_width_mm"] * mm / 2
    h = DIMS["flow_duct_height_mm"] * mm / 2
    lz = min(DIMS["flow_duct_len_mm"] * mm / 2, r_out * 1.9)     # scale length to preview
    yb = -(r_out + 0.16)                                          # underbody, below the engine
    wall = 0.010
    col = (80, 92, 118)
    for sy in (h, -h):                                           # top + bottom walls
        v, f = _box(0.0, yb + sy, zc, w * 2, wall, lz * 2)
        meshes.append(Mesh(v, f, col, group="static"))
    for sx in (w, -w):                                           # left + right walls
        v, f = _box(sx, yb, zc, wall, h * 2, lz * 2)
        meshes.append(Mesh(v, f, col, group="static"))
    zf, zr = zc - lz, zc + lz
    for i in range(6):                                           # front GRILL slats (intake)
        x = -w + (i + 0.5) * (2 * w) / 6.0
        v, f = _box(x, yb, zf, 0.012, h * 1.8, 0.02)
        meshes.append(Mesh(v, f, (120, 132, 155), group="static"))
    for (ww, hh, dz) in [(w * 1.5, h * 1.4, 0.06), (w * 2.0, h * 1.8, 0.15)]:  # rear DIFFUSER
        for sy in (hh, -hh):
            v, f = _box(0.0, yb + sy, zr + dz, ww * 2, wall, 0.02)
            meshes.append(Mesh(v, f, (96, 110, 140), group="static"))
        for sx in (ww, -ww):
            v, f = _box(sx, yb, zr + dz, wall, hh * 2, 0.02)
            meshes.append(Mesh(v, f, (96, 110, 140), group="static"))
    for (dx, dy) in [(-w * 0.5, 0.0), (0.0, 0.0), (w * 0.5, 0.0),
                     (0.0, h * 0.5), (0.0, -h * 0.5)]:            # straight-through flow lines
        meshes.append(_pipe((dx, yb + dy, zf - 0.03), (dx, yb + dy, zr + 0.17),
                            0.004, (150, 210, 255)))
    return meshes


def _hydraulic(zc, r_out):
    """Hydraulic power pack: pump + accumulator + valve block + pressure lines to
    the in-piston wet clutch and the transmission ring-bind passages. Sits at -X."""
    meshes = []
    hx = -(r_out + 0.30)
    v, f = _box(hx, 0.0, zc, 0.12, 0.16, 0.12)          # valve block
    meshes.append(Mesh(v, f, (150, 120, 60), group="static"))
    # pump spins IN PLACE (core-driven) -- built at origin + pivoted, so it does
    # not orbit the engine centre like it did on the mis-placed 'orc' group.
    pv, pf = _solid_cylinder(0.06, -0.05, 0.05, seg=16)
    meshes += _place_spinner([Mesh(pv, pf, (190, 150, 70))],
                             (hx + 0.12, 0.06, zc), (0.0, math.pi / 2), "core")
    v, f = _solid_cylinder(0.07, zc - 0.10, zc + 0.10, seg=16)   # accumulator
    v = np.asarray(v) + np.array([hx, -0.14, 0.0])
    meshes.append(Mesh(v, f, (120, 100, 55), group="static"))
    # pressure lines fanning toward the piston clutch and ring-layer bind passages
    meshes.append(_pipe((hx + 0.06, 0.0, zc), (hx + 0.30, 0.0, zc + 0.2), 0.011, (210, 170, 80)))
    meshes.append(_pipe((hx + 0.06, 0.0, zc), (hx + 0.30, 0.0, zc - 0.2), 0.011, (210, 170, 80)))
    return meshes


def _turbocharger(zc, r_out):
    """Exhaust-driven turbocharger: hot exhaust spins the turbine and the
    compressor feeds intake pressure."""
    meshes = []
    mm = 0.001
    turbine_r = DIMS["turbo_turbine_d_mm"] * mm / 2
    comp_r = DIMS["turbo_comp_d_mm"] * mm / 2
    half_len = DIMS["turbo_len_mm"] * mm / 2
    runner_r = DIMS["turbo_runner_d_mm"] * mm / 2
    a = math.radians(210.0)
    n = np.array([math.cos(a), math.sin(a), 0.0])
    t = np.array([-math.sin(a), math.cos(a), 0.0])
    base = n * (r_out + turbine_r + 0.035) + np.array([0.0, 0.0, zc + 0.02])
    hot = base - t * 0.055
    cold = base + t * 0.060

    # turbine and compressor housings are small compared with the 760 mm rotor.
    v, f = _annulus_cylinder(turbine_r, turbine_r * 0.42,
                             -half_len, half_len, seg=24)
    v = np.asarray(v) @ rot_y(math.pi / 2).T + hot
    meshes.append(Mesh(v, f, (155, 85, 48), group="static", hot=True))
    v, f = _annulus_cylinder(comp_r, comp_r * 0.40,
                             -half_len * 0.88, half_len * 0.88, seg=24)
    v = np.asarray(v) @ rot_y(math.pi / 2).T + cold
    meshes.append(Mesh(v, f, (65, 130, 180), group="static"))
    meshes.append(_pipe(hot + np.array([0.0, 0.0, -0.030]),
                        cold + np.array([0.0, 0.0, -0.030]),
                        0.008, (210, 215, 220)))

    # exhaust runners from the hot rim into the turbine, then away to recovery.
    for off in (-0.032, 0.032):
        rim_pt = n * (r_out * 0.98) + t * off + np.array([0.0, 0.0, zc])
        meshes.append(_pipe(rim_pt, hot - t * 0.060 + np.array([0.0, 0.0, off]),
                            runner_r, (190, 105, 58)))
    meshes.append(_pipe(hot - n * 0.012, hot - n * 0.18,
                        runner_r * 0.82, (170, 90, 55)))

    # compressed intake return to the fixed injector/intake side.
    intake_pt = np.array([0.0, r_out * 1.02, zc + 0.01])
    meshes.append(_pipe(cold + t * 0.055, intake_pt,
                        runner_r * 0.90, (75, 150, 205)))
    v, f = _box(intake_pt[0], intake_pt[1] + 0.035, intake_pt[2],
                0.12, 0.030, 0.035)
    meshes.append(Mesh(v, f, (70, 135, 185), group="static"))
    return meshes


def _automatic_supercharger(zc, r_out):
    """Compact mechanically driven supercharger with an automatic clutch. It
    ramps in slowly when the engine runs and adds intake density/compression."""
    meshes = []
    mm = 0.001
    body_r = DIMS["supercharger_d_mm"] * mm / 2
    half_len = DIMS["supercharger_len_mm"] * mm / 2
    pulley_r = DIMS["supercharger_pulley_d_mm"] * mm / 2
    runner_r = DIMS["turbo_runner_d_mm"] * mm / 2 * 0.82
    a = math.radians(34.0)
    n = np.array([math.cos(a), math.sin(a), 0.0])
    t = np.array([-math.sin(a), math.cos(a), 0.0])
    base = n * (r_out + body_r + 0.050) + np.array([0.0, 0.0, zc + 0.015])
    intake = np.array([0.0, r_out * 1.02, zc + 0.012])

    v, f = _annulus_cylinder(body_r, body_r * 0.42, -half_len, half_len, seg=28)
    v = np.asarray(v) @ rot_y(math.pi / 2).T + base
    meshes.append(Mesh(v, f, (82, 135, 175), group="static"))
    v, f = _solid_cylinder(body_r * 0.48, -half_len * 0.70, half_len * 0.70, seg=20)
    meshes.append(Mesh(v, f, (115, 195, 220), group="super",
                       pivot=base, tilt=(0.0, math.pi / 2)))

    pulley = base - t * 0.080 + np.array([0.0, 0.0, -0.020])
    v, f = _solid_cylinder(pulley_r, -0.014, 0.014, seg=24)
    meshes.append(Mesh(v, f, (185, 175, 110), group="super",
                       pivot=pulley, tilt=(0.0, math.pi / 2)))
    v, f = _annulus_cylinder(pulley_r * 1.12, pulley_r * 0.82, -0.018, 0.018, seg=24)
    meshes.append(Mesh(v, f, (80, 80, 72), group="static",
                       pivot=pulley, tilt=(0.0, math.pi / 2)))

    # Belt/drive line is schematic: it shows the core-driven automatic clutch.
    meshes.append(_pipe((0.0, r_out * 0.42, zc + 0.02), pulley,
                        0.006, (70, 70, 65)))
    meshes.append(_pipe(base + t * 0.055, intake,
                        runner_r, (95, 175, 220)))
    return meshes


def build_engine_parts(n_pistons=1):
    """Construct every mechanical part of the HOHEV-Rotary, to scale (metres),
    grouped into spec'd Parts ordered for assembly. `n_pistons` rotary pistons
    stack on ONE shared core; each piston gets its own wet clutch + injector.
    The transmission lives INSIDE the engine, so each piston stays very thin.
    Every mesh is tagged with its kinematic group so it spins at its true RPM."""
    mm = 0.001
    parts = []
    order = [0]

    def nextord():
        o = order[0]
        order[0] += 1
        return o

    r_out = DIMS["ring_outer_d_mm"] * mm / 2
    th = DIMS["ring_thickness_mm"] * mm
    cth = DIMS["clutch1_thick_mm"] * mm
    fth = DIMS["flywheel_thick_mm"] * mm
    pitch = th + 0.05
    z_front = (n_pistons - 1) * pitch
    z_fly = z_front + th / 2 + cth + fth / 2 + 0.04

    # --- Central shaft (Core) -- shared backbone, assembly anchor ------
    rs = DIMS["shaft_d_mm"] * mm / 2
    v, f = _solid_cylinder(rs, -0.32, z_fly + 0.05, seg=22)
    shaft_m = [Mesh(v, f, C_SHAFT, group="core")]
    shaft_m += _grp(_bolt_ring(rs * 0.62, z_fly + 0.02, 6, 0.006, 0.02, (150, 155, 165)), "core")
    parts.append(Part("shaft", "Central Shaft (Core)", shaft_m,
        ["Function: ONE shared core every piston's clutch engages",
         "Dia %.0f mm  x  spans the whole %d-piston stack" % (DIMS["shaft_d_mm"], n_pistons),
         "Material: forged splined steel",
         "Same core fuelled torque feeds the transmission drive power"],
        order=nextord(), explode=(0, 0, 0.0), color=C_SHAFT))

    # --- Transmission = concentric RING LAYERS (the rings ARE the clutch) --
    # Gear 1 is the center ring. Selecting gear N binds rings 1..N together,
    # while outer rings stay inactive until selected. Each ring has INNER-edge
    # teeth; the small receival gear shifts radially to that selected ring edge.
    n = DIMS["trans_rings"]
    edges = trans_ring_edges()
    ro = edges[-1][1]
    ri = edges[0][0]
    tth = DIMS["trans_thick_mm"] * mm
    zt = -th / 2 - tth / 2 - 0.02
    tmeshes = []
    for k, (inner, outer) in enumerate(edges):
        col = C_TRANS if k % 2 == 0 else C_TRANS_ALT
        v, f = _annulus_cylinder(outer * 0.985, inner, zt - tth / 2, zt + tth / 2, seg=48)
        tmeshes.append(Mesh(v, f, col, group="trans%d" % k))       # each ring its rate
        tooth_count = 14 + k * 4
        teeth = _gear_teeth(inner + 0.006, zt, tooth_count, tth / 2,
                            (185, 150, 55), (0, 0, 0), (0, 0))
        tmeshes += _grp(teeth, "trans%d" % k)
    parts.append(Part("trans", "Transmission Gear Wet-Clutch Ring-Layers (+ output teeth)", tmeshes,
        ["Function: a GEAR WET CLUTCH -- %d concentric RING LAYERS, and each ring" % n,
         "IS its own gear ratio (gear 1 = centre ring, gear %d = outer ring)." % n,
         "The rings BIND together like a multi-plate wet clutch: the TRANSMISSION",
         "BIND slider is that wet-clutch pressure. FULL bind = the selected gear",
         "locks solid; LESS bind = more SLIP (e.g. gear 2 slips more the lower the",
         "bind). A separate SET-GEAR shifter (Gear slider) chooses which ring is",
         "engaged; every ring carries TEETH on its INNER edge and the receival",
         "gear shifts to the selected ring edge. Rings, not discs."],
        order=nextord(), explode=(0, 0, -0.42), color=C_TRANS))

    # --- Axial-flux generator (driven by output gear -> CHARGES) -------
    rg = DIMS["gen_d_mm"] * mm / 2
    gth = DIMS["gen_thick_mm"] * mm
    zg = zt - tth / 2 - gth / 2 - 0.05
    gm = []
    v, f = _solid_cylinder(rg, zg - gth / 2, zg + gth / 2, seg=46)
    gm.append(Mesh(v, f, C_GEN, group="gen"))
    v, f = _annulus_cylinder(rg * 1.06, rg * 0.92, zg - gth / 2 - 0.01, zg + gth / 2 + 0.01, seg=46)
    gm.append(Mesh(v, f, (40, 55, 90), group="static"))
    gm += _grp(_fin_ring(rg * 0.99, zg, 18, (58, 78, 128), fl=0.035, fw=0.013, fz=gth * 0.6), "static")
    parts.append(Part("gen", "Axial-Flux Generator (range-extender charger)", gm,
        ["Function: turns transmission output into electricity -- this output",
         "CHARGES the batteries + capacitors; it does NOT drive the wheels",
         "(battery power energizes the wheel rails; no separate wheel motor).",
         "Dia %.0f mm x %.0f mm axial-flux PM; kW is solved live from" % (
             DIMS["gen_d_mm"], DIMS["gen_thick_mm"]),
         "air mass, fuel flow, shaft power, ratio and generator efficiency."],
        order=nextord(), explode=(0, 0, -1.05), color=C_GEN))

    # --- Toothed receival output gear (meshes trans teeth -> generator) --
    rog = DIMS["out_gear_d_mm"] * mm / 2
    oth = DIMS["out_gear_thick_mm"] * mm
    ang = math.radians(DIMS["out_angle_deg"])
    px = output_gear_mesh_radius(0)
    zog = zt - tth / 2 - oth / 2 - 0.010
    v, f = _solid_cylinder(rog, -oth / 2, oth / 2, seg=28)
    ogm = [Mesh(v, f, C_OUTGEAR, group="out", pivot=(px, 0.0, zog),
                tilt=(ang, 0.0), selectable=True)]
    ogm += _grp(_gear_teeth(rog, 0.0, 18, oth / 2, (185, 150, 55),
                            (px, 0.0, zog), (ang, 0.0), selectable=True), "out")
    # selector rail and detents show the gear shifts to each ring edge
    v, f = _box((output_gear_mesh_radius(0) + output_gear_mesh_radius(n - 1)) / 2,
                -0.085, zog, output_gear_mesh_radius(n - 1) - output_gear_mesh_radius(0),
                0.018, 0.018)
    ogm.append(Mesh(v, f, (120, 130, 145), group="static"))
    for k in range(n):
        v, f = _solid_cylinder(0.008, zog - 0.010, zog + 0.010, seg=8)
        v = np.asarray(v) + np.array([output_gear_mesh_radius(k), -0.085, 0.0])
        ogm.append(Mesh(v, f, (185, 150, 55), group="static"))
    # short visible coupling toward the generator face, moving with the selector
    z0 = zg + gth / 2
    z1 = zog - oth / 2
    v, f = _solid_cylinder(0.008, -abs(z1 - z0) / 2, abs(z1 - z0) / 2, seg=10)
    ogm.append(Mesh(v, f, (205, 210, 220), group="out",
                    pivot=(px, 0.0, (z0 + z1) / 2), selectable=True))
    parts.append(Part("outgear", "Output Receival Gear (-> generator)", ogm,
        ["Function: side-mounted selector pinion shifts to the selected ring's",
         "INNER-edge teeth and drives the generator.",
         "Dia %.0f mm; rotates on its OWN axis like a small gear, ~%.0f deg pivot" % (
             DIMS["out_gear_d_mm"], DIMS["out_angle_deg"]),
         "Unavoidable pivot angle loss taking power off the toothed rim",
         "Feeds the generator to CHARGE batteries/caps -- not the wheels."],
        order=nextord(), explode=(0.95, 0.0, -0.72), color=C_OUTGEAR))

    # --- N rotary pistons, each with an INTEGRAL main clutch + injector -
    for k in range(n_pistons):
        zk = k * pitch
        sfx = "" if n_pistons == 1 else " #%d" % (k + 1)
        ring_part = Part("ring%d" % k, "Rotary Piston%s (8-chamber)" % sfx, _rotary_ring(zk),
            ["Function: circular 'water-wheel' rotary piston on the shared core",
             "OD %.0f / ID %.0f mm, thick %.0f mm  (very thin)" % (
                 DIMS["ring_outer_d_mm"], DIMS["ring_inner_d_mm"], DIMS["ring_thickness_mm"]),
             "%d concave paddles @45 deg; each has ONE fixed combustion area at" % DIMS["chambers"],
             "the rim-top injector -- 1 to 8 combustions per single rotation.",
             "At full lock in high gear, the rotation rate is the limit"],
            order=nextord(), explode=(0, 0, 0.42 + k * 0.42), color=C_RING)
        # fixed ignition area for this piston (rim top, at the injector) -- the
        # combustion flash is drawn here, it does NOT rotate around the ring.
        ring_part.fire_anchor = np.array(
            [r_out * math.cos(INJECTOR_ANGLE), r_out * math.sin(INJECTOR_ANGLE), zk])
        parts.append(ring_part)

        parts.append(Part("clutch1_%d" % k, "Main Wet Clutch%s (multi-plate, in-piston)" % sfx,
            _integral_clutch(zk),
            ["Function: MULTI-PLATE wet clutch integral to the piston. Steel",
             "separator plates splined to the piston's clutch DRUM interleave",
             "with friction plates splined to the central Core hub -- the piston",
             "gives the clutch a real spot to grip onto.",
             "Sits in the piston's inner bore; hydraulic pressure squeezes it.",
             "FULL engagement = 1:1 lock, grabs full piston power (no slip);",
             "less pressure = the plates slip (piston + core faces turn at",
             "different rates). More plates = more controlled slip surface."],
            order=nextord(), explode=(0, 0, 0.28 + k * 0.42), color=C_CLUTCH))

        parts.append(Part("inj%d" % k, "Fuel Injector%s (direct)" % sfx, _injector(zk),
            ["Function: fixed direct-injection nozzle at the rim top",
             "Each chamber gets fuel; optional air-PSI assist is internal only",
             "Drive optimum = low RPM with 8 combustions/rev; slider still tests 1-8",
             "Lean direct injection synced to rotation and chamber position"],
            order=nextord(), explode=(0.0, 0.55, 0.42 + k * 0.42), color=(230, 200, 90)))

        parts.append(Part("compression%d" % k, "Pressure Seal / Air-PSI Gate%s" % sfx,
            _compression_gate(zk),
            ["Function: guarantees seal/charge prep for each selected firing.",
             "The wheel does not need a full piston-style up/down compression",
             "stroke; it needs a reliable airtight pocket at ignition.",
             "Turbo boost substitutes for most physical compression by raising",
             "air density before the chamber reaches the firing position.",
             "Optional air-PSI injectors only add dense air inside the chamber.",
             "The gate mainly seals/holds that pressure and directs swirl so the",
             "shaped pocket catches the full explosion cleanly."],
            order=nextord(), explode=(0.0, 0.66, 0.34 + k * 0.42), color=(80, 155, 135)))

    # --- Hydraulic power pack (main clutch + transmission ring bind) ---
    parts.append(Part("hyd", "Hydraulic Power Pack (clutch actuation)", _hydraulic(0.0, r_out),
        ["Function: pressurizes the main clutch and transmission ring bind",
         "Pump + accumulator + valve block + pressure lines to piston/core",
         "and to the concentric ring layers",
         "More pressure = harder engagement / less slip; low = free slip",
         "Main clutch fully engaged grabs full piston power (no slip)"],
        order=nextord(), explode=(-1.2, 0.0, 0.0), color=(150, 120, 60)))

    # --- Cooling ring around the piston (absorbs heat into the loop) ----
    parts.append(Part("coolring", "Piston Cooling Ring (absorbs heat)",
        _cooling_ring(0.0, r_out),
        ["Function: a slim closed-loop jacket hugging the piston rim that ABSORBS",
         "combustion + friction heat into the working fluid -- the goal is to KEEP",
         "heat, not shed it (~%.0f%% retained). A riser carries the hot fluid down" % (
             THERM["heat_trap_frac"] * 100),
         "to the compact boiler. Kept thin and to scale, not oversized.",
         "Temperature is regulated later by RELEASING pressure, not losing heat."],
        order=nextord(), explode=(0.0, 0.0, 0.55), color=(150, 92, 70)))

    # --- Geometric heat shield (directs heat to the boiler focus) -------
    parts.append(Part("shield", "Geometric Heat Shield (aims heat at boiler)",
        _heat_shield(0.0, r_out),
        ["Function: angled reflector VANES form a funnel that DIRECTS the trapped",
         "heat onto ONE focus area of the boiler -- geometry aims the heat instead",
         "of a metal mass soaking it up. Higher directed coupling = faster boil,",
         "and the block itself holds the heat that keeps the boiler hot after the",
         "engine stops. No heavy heat-absorbing ingots."],
        order=nextord(), explode=(0.0, -0.55, 0.0), color=(170, 176, 186)))

    # --- Compact dual-chamber multi-stage steam pack --------------------
    parts.append(Part("steam", "Steam Boiler + Dual-Chamber Compound Expander",
        _steam_boiler(0.0, r_out),
        ["Function: directed heat boils the closed-loop FLUID; a DUAL-CHAMBER valve",
         "block (two chambers alternating, no dead stroke) feeds a MULTI-STAGE",
         "COMPOUND expander -- pressure is routed, reused stage to stage until weak,",
         "then to a compact GENERATOR. Runs while there is heat AND pressure, even",
         "engine-off. A pressure-RELEASE vent regulates temp by venting PSI (cool by",
         "releasing pressure, not losing heat). %d internal reuse stages." % THERM["compound_stages"],
         "Fluid is selectable (F): non-freezing low-boil fluids (R245FA/ammonia/",
         "methanol) all beat WATER and never freeze in winter -- water freezes at 0C."],
        order=nextord(), explode=(0.0, -1.05, 0.0), color=(70, 150, 210)))

    # --- 2nd-stage exhaust recovery turbine -----------------------------
    parts.append(Part("recov2", "2nd-Stage Exhaust Recovery Turbine",
        _second_stage_turbine(0.0, r_out),
        ["Function: sits DOWNSTREAM of the main turbo and catches MORE of the",
         "leftover exhaust with a 2nd turbine + small generator (~%.0f%% of what" % (
             THERM["second_stage_frac"] * 100),
         "the turbo leaves). The remaining exhaust HEAT is then routed into the",
         "boiler so almost nothing is thrown away.",
         "Turbo -> 2nd turbine -> boiler: two turbines, then heat recovery."],
        order=nextord(), explode=(0.35, -0.55, 0.05), color=(150, 85, 55)))

    # --- Regenerative in-wheel generator --------------------------------
    parts.append(Part("wheelgen", "Rail-Ring Wheel Regen (brake/downhill)",
        _regen_wheel(0.0, r_out),
        ["Function: the road wheels use the same EM rail-ring as a generator.",
         "On braking and downhill the spinning rim back-drives the stationary rail",
         "field and pours recovered momentum straight into supercaps + battery.",
         "This is the primary momentum harvest; the steam loop is the primary",
         "heat harvest. Together they maximize charge with minimal fuel.",
         "Each hub spins on PASSIVE PERMANENT-MAGNET (hard-magnetic Halbach) bearings",
         "-- NOT powered electromagnetic ones -- near-zero friction, no coil power,",
         "diamagnetically stabilized, cutting rolling/parasitic drag ~%.0f%% (info 8c)." % (
             WHEEL_BEARING["roll_cut_frac"] * 100)],
        order=nextord(), explode=(1.35, -0.2, 0.0), color=(70, 95, 150)))

    # --- Passenger pop-out pedal-assist generators ---------------------
    parts.append(Part("pedals", "Passenger Pedal-Assist Generators (%d seats)" % DIMS["seats"],
        _pedal_assist(0.0, r_out),
        ["Function: a POP-OUT foot-pedal generator at EVERY seat. Pushing the",
         "pedals (like pedaling a bike) spins a mini GENERATOR that TRICKLE-charges",
         "the battery -- positionable, and pressable forever for cumulative charge.",
         "~%.0f W per seat x %d seats = up to %.0f W of free human power that adds" % (
             PEDAL_WATTS_PER_SEAT, DIMS["seats"], PEDAL_WATTS_PER_SEAT * DIMS["seats"]),
         "sustain and stretches MPG further. Toggle with K while driving."],
        order=nextord(), explode=(1.7, 0.25, 0.0), color=(200, 190, 90)))

    # --- Multi-layer ambient harvest suite (solar/suspension/TENG/tyre) ---
    parts.append(Part("harvest", "Ambient-Harvest Suite (solar/suspension/TENG)",
        _ambient_harvest(0.0, r_out),
        ["Function: the compounding, ALWAYS-ON, fuel-free trickle -- no single magic",
         "part, every joule attacked from many angles at once:",
         " - SOLAR quantum-dot film over the upper surface (~%.0f W favourable sun)," % AMBIENT_HARVEST_W["solar"],
         " - 4 linear ELECTROMAGNETIC regen suspension dampers (~%.0f W on real roads)," % AMBIENT_HARVEST_W["suspension"],
         " - triboelectric (TENG) underbody film (~%.0f W) + tyre harvesters (~%.0f W)." % (
             AMBIENT_HARVEST_W["triboelectric"], AMBIENT_HARVEST_W["tire"]),
         "~%.0f W EXTERNAL solar charges even coasting engine-OFF; in bright sun below" % (
             (AMBIENT_HARVEST_W["solar"] + AMBIENT_HARVEST_W["solar_roof"]) * SUN),
         "~10-25 mph it can cover the road load (solar-sustained). Core MPG lever."],
        order=nextord(), explode=(0.0, 1.4, 0.0), color=(70, 120, 190)))

    # --- Dedicated PV solar ROOF panel (extra electric) ----------------
    parts.append(Part("solarroof", "Solar Roof Panel (dedicated PV, +electric)",
        _solar_roof(0.0, r_out),
        ["Function: a dedicated high-efficiency PHOTOVOLTAIC ROOF panel ABOVE the",
         "quantum-dot ambient film -- ~%.0f x %.0f mm of extra cells adding up to" % (
             DIMS["solar_roof_len_mm"], DIMS["solar_roof_width_mm"]),
         "~%.0f W of FREE electric straight into the pack in bright sun." % AMBIENT_HARVEST_W["solar_roof"],
         "Because the wheels are ELECTRICALLY driven, every solar watt is traction",
         "the engine never has to make -- always on, even parked, so it lifts MPG.",
         "Stacks with the quantum-dot film for ~%.0f W of total solar." % (
             (AMBIENT_HARVEST_W["solar"] + AMBIENT_HARVEST_W["solar_roof"]) * SUN)],
        order=nextord(), explode=(0.0, 1.9, 0.0), color=(30, 60, 120)))

    # --- Windshield concentrating lens -> boiler (solar HEAT) ----------
    parts.append(Part("sunlens", "Windshield Solar Concentrator Lens (-> boiler)",
        _windshield_concentrator(0.0, r_out),
        ["Function: an AUTO-ADJUSTING Fresnel / magnifying lens (~%.0f mm aperture)" % DIMS["windshield_lens_d_mm"],
         "set into the windshield TRACKS the sun and FOCUSES concentrated sunlight",
         "onto a receiver on the boiler -- dumping up to ~%.2f kW of free HEAT into" % SOLAR_CONCENTRATOR_KW,
         "the closed-loop fluid. The dual-chamber compound expander then makes",
         "ELECTRICITY from sunshine alone, even parked or cruising engine-off.",
         "It DEFOCUSES automatically near the fluid's saturation temp so it never",
         "over-pressures. Works best in bright HOT sun -- exactly when you drive in",
         "the heat -- so hot sunny days add power and MPG instead of just heating you."],
        order=nextord(), explode=(0.0, 1.2, 0.9), color=(255, 210, 90)))

    # --- Super-hydrophobic + plastron slip body skin (drag cut) --------
    parts.append(Part("hydroskin", "Super-Hydrophobic Slip Skin (humid-air drag cut)",
        _hydro_skin(0.0, r_out),
        ["Function: a nano-textured SUPER-HYDROPHOBIC coating over the whole body.",
         "Rain + humidity BEAD UP and roll straight off instead of forming a clinging",
         "water FILM -- so in wet/humid air the car stops dragging (and carrying) a",
         "skin of water, a real drag + weight penalty on ordinary paint.",
         "The same micro-texture traps a thin AIR 'plastron' layer -- a near-",
         "frictionless, VACUUM-LIKE slip boundary so the airflow rides on air, not",
         "on the paint, cutting turbulent SKIN-FRICTION drag even in dry air",
         "(the shark-skin riblet effect).",
         "Effect: effective Cd cut ~%.1f%% dry, up to ~%.1f%% in humid air (now RH" % (
             HYDRO["dry_frac"] * 100, (1.0 - hydro_cd_factor()) * 100),
         "%2.0f%%). Self-cleaning + anti-icing too. Pure drag reduction = higher MPG." % (
             HUMIDITY * 100)],
        order=nextord(), explode=(0.95, 1.15, 0.0), color=(150, 205, 235)))

    # --- Passive drag-air capture skin + recovery turbine --------------
    parts.append(Part("dragcapture", "Passive Drag-Air Capture Turbine (recover drag)",
        _drag_groove_skin(0.0, r_out),
        ["Function: the hydrophobic skin makes MOST air slip, but the little that",
         "still grips the body is caught by directional MICRO-GROOVES (deep shark-skin",
         "riblets aligned WITH the airflow) that funnel it through internal ducts into",
         "a tiny low-drag BLADELESS (Tesla-type) turbine + generator. Fully PASSIVE --",
         "no pumps or valves. It turns a slice of otherwise-WASTED drag energy into",
         "electricity, recovering ~%.0f%% of the remaining aero drag power (groove %.0f%%" % (
             DRAG_CAPTURE["groove_eff"] * DRAG_CAPTURE["turbine_eff"] * 100,
             DRAG_CAPTURE["groove_eff"] * 100),
         "x turbine %.0f%%). It scales with speed (aero power ~ v^3), so it gives the" % (
             DRAG_CAPTURE["turbine_eff"] * 100),
         "most at highway speed, and always recovers LESS than the drag (no free",
         "energy) -- it simply stops wasting it. Works with the skin, not against it."],
        order=nextord(), explode=(1.5, -0.3, 0.0), color=(120, 175, 215)))

    # --- Deployable parked windmill / tail fin -------------------------
    parts.append(Part("windmill", "Parked Windmill / Deployable Tail Fin (wind charge)",
        _windmill_fin(0.0, r_out),
        ["Function: when the car is PARKED (or crawling), a rear tail fin EXTENDS and",
         "works as a small VERTICAL-AXIS wind turbine, trickle-charging the battery",
         "from ambient WIND alongside the solar roof -- free range with zero driver",
         "input over hours/days. Up to ~%.0f W in strong wind (~%.2f m^2 capture)." % (
             WINDMILL["max_w"], WINDMILL["area_m2"]),
         "It RETRACTS above ~%.0f mph while driving, where it would only add drag." % WINDMILL["park_mph"],
         "Deploys passively (shape-memory / small servo on an anemometer trigger).",
         "Combined with solar it keeps the pack topped for the next drive for free."],
        order=nextord(), explode=(0.0, 0.4, -1.5), color=(150, 205, 235)))

    # --- Through-body straight flow-through duct (rear-wake drag kill) --
    parts.append(Part("flowduct", "Through-Body Flow Duct (kills rear-wake drag)",
        _flow_through_duct(0.0, r_out),
        ["Function: a wide, thin, STRAIGHT hollow duct from a low front-GRILL intake,",
         "straight through the body, to a rear DIFFUSER. High-pressure nose air flows",
         "100%% straight through the car and exits into the low-pressure WAKE, FILLING",
         "the rear vacuum pocket -- so it attacks PRESSURE (form) drag, the dominant",
         "drag source above ~50 mph. The rear diffuser widens like a venturi to slow",
         "the exit air smoothly and recover pressure.",
         "Scaled straight-through pipe (%.0f x %.0f mm, ~%.1f m long) with front slots" % (
             DIMS["flow_duct_width_mm"], DIMS["flow_duct_height_mm"],
             DIMS["flow_duct_len_mm"] / 1000.0),
         "and a rear diffuser. It cuts the effective Cd ~%.0f-%.0f%% (stronger at speed)," % (
             DUCT["base"] * DUCT["blend"] * 100,
             (DUCT["base"] + DUCT["speed"]) * DUCT["blend"] * 100),
         "pairing with the boat-tail, hydrophobic skin and grooves so the car behaves",
         "like a near-perfect streamlined body. Parked, it doubles as passive cabin/",
         "battery ventilation."],
        order=nextord(), explode=(0.0, -0.9, 1.6), color=(100, 120, 150)))

    # --- Exhaust turbocharger accessory pack ---------------------------
    parts.append(Part("turbo", "Exhaust Turbocharger Unit", _turbocharger(0.0, r_out),
        ["Function: exhaust-driven turbo feeds baseline intake density.",
         "Hot exhaust enters the turbine; the compressor returns boosted intake",
         "air to the injector/intake side.",
         "Scaled %.0f mm turbine / %.0f mm compressor, mounted at the rim." % (
             DIMS["turbo_turbine_d_mm"], DIMS["turbo_comp_d_mm"]),
         "Pressure ratio feeds the live air/fuel physics and generator kW."],
        order=nextord(), explode=(0.45, -0.25, 0.04), color=(85, 145, 190)))

    # --- Slow automatic supercharger -----------------------------------
    parts.append(Part("supercharger", "Automatic Supercharger Unit",
        _automatic_supercharger(0.0, r_out),
        ["Function: compact mechanical supercharger with automatic clutch.",
         "It engages slowly only while the engine is running, increasing air",
         "density/compression without replacing the circular piston shape.",
         "The internal air-PSI jets remain optional and only inside the chamber.",
         "Engagement ramps in the live readout; full assist adds ~%.0f%% boost." % (
             SUPERCHARGER_BOOST_GAIN * 100)],
        order=nextord(), explode=(0.58, 0.36, 0.08), color=(82, 135, 175)))

    # --- Tungsten kinetic flywheel (free-float, very front) ------------
    rf = DIMS["flywheel_d_mm"] * mm / 2
    fm = []
    v, f = _solid_cylinder(rf, z_fly - fth / 2, z_fly + fth / 2, seg=52)
    fm.append(Mesh(v, f, C_FLYWHEEL, group="fly"))
    v, f = _annulus_cylinder(rf * 0.45, rf * 0.30, z_fly - fth / 2 - 0.005, z_fly + fth / 2 + 0.005, seg=28)
    fm.append(Mesh(v, f, (90, 40, 38), group="fly"))
    fm += _grp(_bolt_ring(rf * 0.78, z_fly, 8, 0.012, fth * 0.55, (60, 28, 26)), "fly")
    ifw = DIMS["flywheel_mass_kg"] * rf * rf
    parts.append(Part("flywheel", "Tungsten Kinetic Flywheel (on core)", fm,
        ["Function: mounted ON the central core; stores kinetic energy.",
         "When the MAIN CLUTCH disengages the disc FREE-SPINS with its stored",
         "energy -- the transmission can take that energy, or use it to reduce",
         "stall. Dia %.0f mm x %.0f mm, %.0f kg tungsten." % (
             DIMS["flywheel_d_mm"], DIMS["flywheel_thick_mm"], DIMS["flywheel_mass_kg"]),
         "Inertia ~%.2f kg.m^2 ; ~1.9 kWh @ %.0f rpm; PASSIVE permanent-magnet" % (ifw, FLYWHEEL_MAX_RPM),
         "(not powered electromagnetic) bearings, diamagnetically stabilized (info 8c)"],
        order=nextord(), explode=(0, 0, 0.42 + n_pistons * 0.42 + 0.12), color=C_FLYWHEEL))

    return parts


def build_car_parts():
    """Construct the ENTIRE VEHICLE to real-life scale (metres) from CAR/DIMS, as
    spec'd Parts for the separate CAR VIEW: a teardrop carbon monocoque body, glass
    canopy, structural battery floor, 4 rail-ring direct-drive road wheels, the rotary
    range-extender engine + tungsten flywheel, the steam boiler, the solar roof, the
    windshield sun-lens, the through-body flow duct, the parked windmill fin, the
    regen suspension dampers, the passenger pedal generators and the supercap bank.
    Every part carries real dimensions in its spec and an explode offset for the
    whole-car exploded view. Non-spinning parts use the 'static' group; wheels/engine/
    flywheel/windmill/pedals spin on their live kinematic groups."""
    mm = 0.001
    L = CAR["length_mm"] * mm
    W = CAR["width_mm"] * mm
    Ht = CAR["height_mm"] * mm
    wb = CAR["wheelbase_mm"] * mm
    wr = CAR["wheel_d_mm"] * mm / 2
    ww = CAR["wheel_width_mm"] * mm
    gc = CAR["ground_clear_mm"] * mm
    half = L / 2
    axf, axr = wb / 2, -wb / 2
    tx = CAR["track_mm"] * mm / 2

    parts = []
    order = [0]

    def nextord():
        o = order[0]
        order[0] += 1
        return o

    def static(v, f, col):
        return Mesh(v, f, col, group="static")

    # --- Carbon monocoque BODY (teardrop + boat-tail), lofted to scale -----
    # Control sections (z, halfwidth, y_bottom, y_top); Catmull-Rom subdivided for
    # a smooth shell that still passes through each control station.
    body_ctrl = [
        (-half, 0.30, 0.30, 0.55), (-1.40, 0.56, 0.20, 0.86),
        (-0.70, 0.74, 0.16, 1.06), (0.00, 0.80, 0.14, Ht),
        (0.70, 0.75, 0.14, 1.02), (1.30, 0.60, 0.16, 0.72),
        (1.72, 0.40, 0.20, 0.52), (half, 0.22, 0.24, 0.42),
    ]
    body_sections = _smooth_sections(body_ctrl, sub=3)
    bv, bf = _hull(body_sections)
    body = [static(bv, bf, (46, 110, 175))]
    # sealed flat underbody panel
    uv, uf = _box(0.0, gc, -0.1, W * 0.9, 0.02, L * 0.82)
    body.append(static(uv, uf, (30, 40, 56)))
    parts.append(Part("body", "Carbon Monocoque Body + Hydrophobic Skin", body,
        ["Function: the AI-optimized TEARDROP + boat-tail carbon monocoque shell that",
         "sets the road load. Length %.0f / width %.0f / height %.0f mm, wheelbase" % (
             CAR["length_mm"], CAR["width_mm"], CAR["height_mm"]),
         "%.0f mm, sealed flat underbody at %.0f mm ride height." % (
             CAR["wheelbase_mm"], CAR["ground_clear_mm"]),
         "Cd ~%.2f, frontal area ~%.2f m^2 (= %.2f x %.2f m x %.2f fill), curb ~%.0f kg." % (
             VEH["Cd"], VEH["frontal_area_m2"], W, Ht, CAR["frontal_fill"], VEH["curb_mass_kg"]),
         "Wears the SUPER-HYDROPHOBIC + plastron slip skin (info 8b) and the through-",
         "body flow duct + drag-capture grooves -- so most air slips, the rest is",
         "caught, and the wake is filled. Structural battery is integrated in the floor."],
        order=nextord(), explode=(0.0, 0.0, 0.0), color=(46, 110, 175)))

    # --- Glass canopy / cabin --------------------------------------------
    canopy_ctrl = [
        (-0.45, 0.40, 0.80, 1.02), (0.00, 0.52, 0.82, Ht - 0.02),
        (0.55, 0.46, 0.80, 1.00), (0.95, 0.30, 0.78, 0.86),
    ]
    cv, cf = _hull(_smooth_sections(canopy_ctrl, sub=3))
    parts.append(Part("canopy", "Tandem Cabin + Bubble Canopy", [static(cv, cf, (150, 200, 235))],
        ["Function: the low, narrow tandem cabin under a bubble canopy -- minimal",
         "frontal area for a real occupant layout. Camera mirrors (no drag pods).",
         "Seats the driver + passengers who can add free PEDAL power on long trips."],
        order=nextord(), explode=(0.0, 1.4, 0.0), color=(150, 200, 235)))

    # --- Structural battery floor + supercap bank ------------------------
    bv, bf = _box(0.0, gc + 0.07, -0.1, W * 0.84, 0.13, L * 0.60)
    parts.append(Part("battery", "Structural Battery Pack (floor)", [static(bv, bf, (54, 64, 88))],
        ["Function: the ~%.0f kWh structural battery integrated INTO the monocoque" % ELEC["batt_kwh"],
         "floor -- it IS part of the structure, saving weight. Held in a %.0f-%.0f%%" % (
             ELEC["soc_min"] * 100, ELEC["soc_max"] * 100),
         "SOC window for maximum regen headroom. Feeds the 4 direct wheel rails; it is",
         "charged by the rotary generator, regen, steam, solar and all the harvesters."],
        order=nextord(), explode=(0.0, -1.6, 0.0), color=(54, 64, 88)))
    sv, sf = _box(0.0, gc + 0.10, -0.95, W * 0.5, 0.16, 0.34)
    parts.append(Part("supercap", "Supercapacitor Bank", [static(sv, sf, (70, 120, 150))],
        ["Function: a ~%.2f kWh supercapacitor bank that swallows and delivers every" % ELEC["supercap_kwh"],
         "fast charge + drain spike (braking, downhill, engine bursts) so the battery",
         "sees only smooth current -- key to the ~%.0f%% rail-regen recovery." % (
             VEH["regen_frac"] * 100)],
        order=nextord(), explode=(-1.6, -0.4, -0.3), color=(70, 120, 150)))

    # --- 4 road wheels: EM RAIL-RING direct drive (rim IS the rotor) ------
    def wheel(px, pz, axle, side):
        ms = []
        piv = (px, wr, pz)
        short = ("%s%s" % ("F" if axle == "front" else "R",
                           "R" if side == "right" else "L"))
        detail = (axle == "front" and side == "right")

        def lname(text):
            return ("%s %s" % (short, text)) if detail else ""

        def static_ring(v, f, col, name=""):
            vv = np.asarray(v) @ rot_y(math.pi / 2).T + np.array(piv)
            return Mesh(vv, f, col, name=name, group="static")

        # OUTER-ROTOR IN-WHEEL DRIVE (to scale): everything lives INSIDE the tyre
        # radius, so nothing pokes below the road or past the body. The rim IS the
        # rotor; a fixed EM rail stator sits within it across the precision air gap.
        # TYRE: low-profile tread + sidewall. Outer radius = wr (touches the road).
        tv, tf = _annulus_cylinder(wr, wr * 0.80, -ww / 2, ww / 2, seg=28)
        ms += _place_spinner([Mesh(tv, tf, (20, 20, 24))],
                             piv, (0.0, math.pi / 2), "wheel")

        # Structural RIM = rotor shell (spins with the wheel).
        rv, rf = _annulus_cylinder(wr * 0.80, wr * 0.70, -ww * 0.42, ww * 0.42, seg=26)
        ms += _place_spinner([Mesh(rv, rf, (188, 194, 205),
                                   name=lname("rim (rotor shell)"))],
                             piv, (0.0, math.pi / 2), "wheel")
        # Bonded rotor MAGNET BAND on the rim inner face (spins WITH the wheel).
        mv, mf = _annulus_cylinder(wr * 0.70, wr * 0.64, -ww * 0.34, ww * 0.34, seg=28)
        ms += _place_spinner([Mesh(mv, mf, (155, 70, 175),
                                   name=lname("bonded rotor magnet band"))],
                             piv, (0.0, math.pi / 2), "wheel")
        # Visible in-rim rotor / flywheel disc between rim and hub.
        dv, df = _annulus_cylinder(wr * 0.60, wr * 0.26, -ww * 0.14, ww * 0.14, seg=28)
        ms += _place_spinner([Mesh(dv, df, (140, 148, 164),
                                   name=lname("in-rim rotor disc"))],
                             piv, (0.0, math.pi / 2), "wheel")
        for a in range(6):                                   # stiff visible spokes
            an = a * 2 * math.pi / 6
            sv, sf = _box(wr * 0.44 * math.cos(an), wr * 0.44 * math.sin(an), 0.0,
                          wr * 0.50, 0.030, ww * 0.34)
            ms += _place_spinner([Mesh(sv, sf, (108, 115, 130))],
                                 piv, (0.0, math.pi / 2), "wheel")

        # Passive permanent-magnet BEARING HUB (center).
        bv, bf = _annulus_cylinder(wr * 0.24, wr * 0.10, -ww * 0.30, ww * 0.30, seg=20)
        ms += _place_spinner([Mesh(bv, bf, (95, 205, 150),
                                   name=lname("PM bearing hub"))],
                             piv, (0.0, math.pi / 2), "wheel")

        # STATIONARY EM RAIL STATOR: fixed ring INSIDE the rotor magnet band across
        # the precision air gap. It does NOT spin; rail current turns the rim rotor.
        gap = max(0.004, RAIL_DRIVE["air_gap_mm"] * 0.001 * 20.0)   # gap exaggerated to read
        stat_o = wr * 0.64 - gap
        gv, gf = _annulus_cylinder(wr * 0.64, stat_o, -ww * 0.32, ww * 0.32, seg=28)
        ms.append(static_ring(gv, gf, (40, 170, 210), lname("%.2f mm rail air gap" %
                                                            RAIL_DRIVE["air_gap_mm"])))
        sv, sf = _annulus_cylinder(stat_o, wr * 0.40, -ww * 0.30, ww * 0.30, seg=28)
        ms.append(static_ring(sv, sf, (55, 105, 190), lname("stationary EM rail stator")))
        cv, cf = _annulus_cylinder(stat_o - 0.005, stat_o - 0.026, -ww * 0.24, ww * 0.24, seg=26)
        ms.append(static_ring(cv, cf, (190, 120, 45), lname("copper rail winding")))
        for a in range(24):                                  # visible markers for 96 poles
            an = a * 2 * math.pi / 24
            by = wr + (stat_o - 0.014) * math.sin(an)
            bz = pz + (stat_o - 0.014) * math.cos(an)
            pv, pf = _box(px, by, bz, ww * 0.48, 0.013, 0.013)
            ms.append(Mesh(pv, pf, (100, 155, 225), group="static"))
        return ms

    wheel_defs = [
        (tx, axf, "front", "right"),
        (-tx, axf, "front", "left"),
        (tx, axr, "rear", "right"),
        (-tx, axr, "rear", "left"),
    ]
    for px, pz, axle, side in wheel_defs:
        title = "%s %s EM Rail-Ring Wheel Drive" % (axle.title(), side.title())
        ex = (0.72 if side == "right" else -0.72,
              -1.20,
              0.48 if axle == "front" else -0.48)
        parts.append(Part("wheel_%s_%s" % (axle[0], side[0]), title,
            wheel(px, pz, axle, side),
            ["Function: one complete outer-rotor in-wheel drive module; the %.0f mm road" % CAR["wheel_d_mm"],
             "wheel contains the rim rotor shell, bonded rotor magnet band, in-rim rotor/",
             "flywheel disc, copper winding, passive permanent-magnet bearing hub and the",
             "fixed EM rail stator. The stationary %.0f-pole rail stator sits WITHIN the rim" % RAIL_DRIVE["poles"],
             "across a %.2f mm air gap; the rim/disc assembly IS the rotor and spins around" % RAIL_DRIVE["air_gap_mm"],
             "it, so there is no driveshaft, gearbox or wheel clutch. Rail current 0-%.0f A" % RAIL_DRIVE["coil_amp_max"],
             "drives the wheel and the same stator regenerates on braking; all four modules",
             "share the ~%.0f kW / %.0f%% regen system." % (
                 VEH["max_rail_kw"], VEH["regen_frac"] * 100)],
            order=nextord(), explode=ex, color=(70, 95, 150)))

    # --- Wheel-arch fenders (body-coloured flares over each tyre) ---------
    fenders = []
    for px, pz, axle, side in wheel_defs:
        av, af = _arc_shell(wr + 0.060, wr + 0.004, -ww * 0.60, ww * 0.60,
                            math.radians(16), math.radians(164), seg=16)
        vv = np.asarray(av) @ rot_y(math.pi / 2).T + np.array([px, wr, pz])
        fenders.append(Mesh(vv, af, (40, 96, 155), group="static"))
    parts.append(Part("fenders", "Wheel-Arch Fenders", fenders,
        ["Function: the four body-coloured wheel-arch fenders that fair the tops of",
         "the tyres into the monocoque shell -- they seal the wheel wells (part of the",
         "low-Cd covered-wheel aero) and finish the body around each rail-ring drive."],
        order=nextord(), explode=(0.0, 1.2, 0.0), color=(40, 96, 155)))

    # --- EM rail-ring drive unit (standalone showcase, cross-section) ----
    rmc = np.array([0.0, 0.30, -0.10])
    rd = 0.22
    rm = []
    dv, df = _solid_cylinder(rd, -0.06, 0.06, seg=26)         # main disc (rotor + flywheel)
    rm += _place_spinner([Mesh(dv, df, (140, 150, 165))], tuple(rmc), (0.0, math.pi / 2), "core")
    iv, iff = _solid_cylinder(rd * 0.42, -0.075, 0.075, seg=18)  # inner kinetic-store hub
    rm += _place_spinner([Mesh(iv, iff, (110, 60, 55))], tuple(rmc), (0.0, math.pi / 2), "core")
    sv, sf = _annulus_cylinder(rd + 0.045, rd + 0.012, -0.075, 0.075, seg=30)  # EM rail stator
    sv = np.asarray(sv) @ rot_y(math.pi / 2).T + rmc
    rm.append(Mesh(sv, sf, (70, 95, 150), group="static"))
    for a in range(16):                                      # rail poles
        an = a * 2 * math.pi / 16
        by = rmc[1] + (rd + 0.028) * math.sin(an)
        bz = rmc[2] + (rd + 0.028) * math.cos(an)
        pv, pf = _box(rmc[0], by, bz, 0.11, 0.02, 0.02)
        rm.append(Mesh(pv, pf, (95, 125, 185), group="static"))
    cv, cf = _solid_cylinder(rd * 0.6, 0.065, 0.10, seg=20)   # single-plate wet clutch
    cv = np.asarray(cv) @ rot_y(math.pi / 2).T + rmc
    rm.append(Mesh(cv, cf, (60, 140, 165), group="static"))
    rm.append(_pipe((rmc[0] - 0.20, rmc[1], rmc[2]), (rmc[0] - 0.02, rmc[1], rmc[2]),
                    0.02, (200, 205, 215)))                  # input power shaft
    rm.append(_pipe((rmc[0] + 0.10, rmc[1], rmc[2]), (rmc[0] + 0.26, rmc[1], rmc[2]),
                    0.016, (200, 205, 215)))                 # output shaft
    parts.append(Part("raildrive", "EM Rail-Ring Drive Unit (standalone cross-section)", rm,
        ["Function: a standalone cross-section of the direct rail-disc principle.",
         "A stationary EM RAIL surrounds the disc/rim; energising it magnetically",
         "SPINS + LOCKS the rotor with scalable torque (rail current 0-%.0f A)." % RAIL_DRIVE["coil_amp_max"],
         "The disc is a FLYWHEEL kinetic store (inner rim to ~%.0f rpm) and clutches to the output" % RAIL_DRIVE["flywheel_rpm"],
         "via a single-plate wet clutch. Standalone unit: %.1f m disc, ~%.0f kg," % (
             RAIL_DRIVE["disc_dia_m"], RAIL_DRIVE["unit_kg"]),
         "~%.0fk Nm continuous / ~%.0fk Nm burst, %.1f%% efficient, %.2f mm air gap." % (
             RAIL_DRIVE["cont_torque_nm"] / 1000, RAIL_DRIVE["burst_torque_nm"] / 1000,
             RAIL_DRIVE["eff"] * 100, RAIL_DRIVE["air_gap_mm"]),
         "In the car it is the light rim-rotor derivative driving each wheel (info 11)."],
        order=nextord(), explode=(1.6, 0.2, 0.0), color=(90, 120, 180)))

    # --- Rotary range-extender ENGINE + tungsten flywheel ----------------
    ecen = (0.0, 0.40, -1.30)
    eng = []
    ev, ef = _solid_cylinder(0.22, -0.20, 0.20, seg=26)
    eng += _place_spinner([Mesh(ev, ef, (96, 110, 128))], ecen, (0.0, math.pi / 2), "core")
    for a in range(8):                                       # chamber paddles hint
        an = a * 2 * math.pi / 8
        pv, pf = _box(0.17 * math.cos(an), 0.17 * math.sin(an), 0.0, 0.05, 0.05, 0.34)
        eng += _place_spinner([Mesh(pv, pf, (70, 78, 92))], ecen, (0.0, math.pi / 2), "core")
    parts.append(Part("engine", "Rotary Range-Extender Engine (8-chamber)", eng,
        ["Function: the 8-chamber CIRCULAR rotary combustion ring -- the ONLY fuel",
         "burner. It is a range-extender: it runs in short lean bursts to make",
         "electricity, never driving the wheels. See the ENGINE PREVIEW for the full",
         "mechanical detail (clutch, transmission, generator, heat recovery).",
         "Fires only when needed; target ~%.1f%% fuel-to-wheel." % (FUEL_TO_WHEEL_EFF * 100)],
        order=nextord(), explode=(0.0, -0.5, -1.9), color=(96, 110, 128)))
    fv, ff = _solid_cylinder(0.16, -0.05, 0.05, seg=22)
    fly = _place_spinner([Mesh(fv, ff, C_FLYWHEEL)], (0.0, 0.40, -1.02), (0.0, math.pi / 2), "fly")
    parts.append(Part("flywheel", "Tungsten Kinetic Flywheel", fly,
        ["Function: the %.0f kg tungsten flywheel on the engine core. FREE-SPINS when" % DIMS["flywheel_mass_kg"],
         "the main clutch disengages, storing downhill/coast/regen momentum on passive",
         "magnetic bearings. Winds up on descents with the combustion ring OFF."],
        order=nextord(), explode=(0.0, 0.5, -2.3), color=C_FLYWHEEL))

    # --- Steam boiler / heat-recovery drum (tucked beside the engine) -----
    bcen = np.array([0.30, 0.34, -1.35])
    bv, bf = _solid_cylinder(0.10, -0.15, 0.15, seg=18)
    bv = np.asarray(bv) @ rot_y(math.pi / 2).T + bcen
    parts.append(Part("boiler", "Steam Boiler + Heat-Recovery Loop", [static(bv, bf, (70, 150, 210))],
        ["Function: the closed-loop AMMONIA boiler + dual-chamber compound expander",
         "that turns trapped engine heat (and windshield sun-lens heat, and PCM-banked",
         "heat) into electricity -- and keeps charging AFTER the engine stops.",
         "PCM heat bank + thermoelectrics ride alongside it (info 6f)."],
        order=nextord(), explode=(1.6, 0.3, -1.4), color=(70, 150, 210)))

    # --- Solar ROOF panel ------------------------------------------------
    rv, rf = _box(0.0, Ht + 0.015, 0.12, W * 0.64, 0.02, L * 0.44)
    roof = [static(rv, rf, (18, 34, 78))]
    for gx in range(-3, 4):
        x = gx * (W * 0.64) / 7.0
        m = _pipe((x, Ht + 0.03, -0.8), (x, Ht + 0.03, 1.1), 0.004, (150, 160, 185))
        roof.append(m)
    parts.append(Part("solarroof", "Solar Roof Panel (PV) + Quantum-Dot Film", roof,
        ["Function: the high-eff PHOTOVOLTAIC roof panel + quantum-dot film over the",
         "whole upper surface -- up to ~%.0f W of free electric straight to the pack" % (
             (AMBIENT_HARVEST_W["solar"] + AMBIENT_HARVEST_W["solar_roof"]) * SUN),
         "in good sun (info 6g). Charges even parked. The wheels are electric, so",
         "every solar watt is traction the engine never has to make."],
        order=nextord(), explode=(0.0, 1.9, 0.2), color=(30, 60, 120)))

    # --- Windshield SUN-concentrator lens --------------------------------
    lc = np.array([0.0, 0.86, 0.90])
    R = rot_x(-math.radians(55.0))
    lv, lf = _solid_cylinder(0.13, -0.01, 0.01, seg=24)
    lv = np.asarray(lv) @ R.T + lc
    parts.append(Part("sunlens", "Windshield Solar Concentrator Lens", [static(lv, lf, (170, 210, 235))],
        ["Function: the auto-adjusting Fresnel / magnifying lens in the windshield",
         "that focuses sun onto the boiler receiver, adding up to ~%.2f kW of free" % SOLAR_CONCENTRATOR_KW,
         "HEAT into the steam loop -- more power on hot sunny days (info 6g)."],
        order=nextord(), explode=(0.0, 1.4, 1.6), color=(255, 210, 90)))

    # --- Through-body straight FLOW DUCT ---------------------------------
    yb, dw, dh, wall = gc + 0.03, 0.32, 0.09, 0.012
    zf, zr = 1.60, -1.60
    duct = []
    for sy in (dh, -dh):
        v, f = _box(0.0, yb + sy, 0.0, dw * 2, wall, (zf - zr))
        duct.append(static(v, f, (80, 92, 118)))
    for sx in (dw, -dw):
        v, f = _box(sx, yb, 0.0, wall, dh * 2, (zf - zr))
        duct.append(static(v, f, (80, 92, 118)))
    for i in range(6):                                       # front grill slats
        x = -dw + (i + 0.5) * (2 * dw) / 6.0
        v, f = _box(x, yb, zf, 0.014, dh * 1.7, 0.03)
        duct.append(static(v, f, (120, 132, 155)))
    for (dx, dy) in [(-dw * 0.5, 0), (0, 0), (dw * 0.5, 0), (0, dh * 0.5), (0, -dh * 0.5)]:
        duct.append(_pipe((dx, yb + dy, zf + 0.05), (dx, yb + dy, zr - 0.2), 0.004, (150, 210, 255)))
    parts.append(Part("flowduct", "Through-Body Flow Duct (kills rear-wake drag)", duct,
        ["Function: the wide, thin, STRAIGHT hollow duct from the front grill straight",
         "through the car to a rear diffuser -- nose air flows through and FILLS the",
         "rear wake, killing pressure drag (dominant at speed, info 8d).",
         "Cuts the effective Cd ~%.0f-%.0f%% (stronger at speed). Parked, it doubles" % (
             DUCT["base"] * DUCT["blend"] * 100, (DUCT["base"] + DUCT["speed"]) * DUCT["blend"] * 100),
         "as passive cabin/battery ventilation."],
        order=nextord(), explode=(0.0, -1.9, 0.0), color=(100, 120, 150)))

    # --- Parked windmill / deployable tail fin ---------------------------
    wbase = np.array([0.0, 0.48, -1.72])
    wm = []
    mv, mf = _solid_cylinder(0.018, 0.0, 0.22, seg=12)
    mv = np.asarray(mv) @ rot_x(-math.pi / 2).T + wbase
    wm.append(static(mv, mf, (120, 130, 145)))
    fvv, fff = _box(0.0, 0.11, -0.03, 0.010, 0.18, 0.12)
    fvv = np.asarray(fvv) + wbase
    wm.append(static(fvv, fff, (90, 110, 140)))
    whub = wbase + np.array([0.0, 0.24, 0.0])
    for k in range(3):
        a = k * 2 * math.pi / 3
        bvv, bff = _box(0.055 * math.cos(a), 0.055 * math.sin(a), 0.0, 0.03, 0.03, 0.11)
        wm += _place_spinner([Mesh(bvv, bff, (150, 205, 235))], tuple(whub), (math.pi / 2, 0.0), "windmill")
    parts.append(Part("windmill", "Parked Windmill / Deployable Tail Fin", wm,
        ["Function: the rear tail fin that EXTENDS when parked and works as a small",
         "vertical-axis wind turbine, trickle-charging from ambient wind (up to ~%.0f W)" % WINDMILL["max_w"],
         "alongside the solar roof -- free range with zero driver input (info 6h).",
         "Retracts while driving (where it would only be drag)."],
        order=nextord(), explode=(0.0, 1.0, -2.3), color=(150, 205, 235)))

    # --- Regenerative electromagnetic suspension dampers (x4) ------------
    damp = []
    for px in (tx, -tx):
        for pz in (axf, axr):
            base = np.array([px * 0.80, wr + 0.02, pz])
            dv, df = _solid_cylinder(0.032, -0.13, 0.13, seg=12)
            dv = np.asarray(dv) @ rot_x(-math.pi / 2).T + base   # stand the damper upright (Y)
            damp.append(static(dv, df, (86, 108, 150)))
            m = _pipe(tuple(base + np.array([0.0, 0.13, 0.0])),
                      (px * 0.5, gc + 0.14, pz), 0.006, (90, 200, 255))
            damp.append(m)
    parts.append(Part("suspension", "Regenerative EM Suspension Dampers (x4)", damp,
        ["Function: four linear ELECTROMAGNETIC regenerative dampers -- they control",
         "the ride AND harvest road bumps (up to ~%.0f W extra on rough/graded roads" % SUSP_BUMP_W_MAX,
         "over the ~%.0f W baseline). Every bump becomes charge instead of heat." % AMBIENT_HARVEST_W["suspension"]],
        order=nextord(), explode=(1.7, -0.3, 0.0), color=(86, 108, 150)))

    # --- Passenger pedal generators --------------------------------------
    ped = []
    pcen = (0.0, 0.34, 0.55)
    pv, pf = _solid_cylinder(0.06, -0.02, 0.02, seg=14)
    ped += _place_spinner([Mesh(pv, pf, (150, 160, 175))], pcen, (0.0, math.pi / 2), "pedal")
    for a in (0.0, math.pi):
        av, af = _box(0.07 * math.cos(a), 0.07 * math.sin(a), 0.0, 0.09, 0.02, 0.02)
        ped += _place_spinner([Mesh(av, af, (200, 190, 90))], pcen, (0.0, math.pi / 2), "pedal")
    parts.append(Part("pedals", "Passenger Pedal Generators", ped,
        ["Function: pop-out foot-pedal generators (one per seat, %d seats). Passengers" % DIMS["seats"],
         "pedal like a bike to TRICKLE-charge the pack -- ~%.0f W/seat of free human" % PEDAL_WATTS_PER_SEAT,
         "power, cumulative over a trip. Toggle in DRIVE with K (info 6b)."],
        order=nextord(), explode=(1.5, 0.3, 0.6), color=(200, 190, 90)))

    # --- Inertial pendulum rim (lower bumper-skirt harvester) ------------
    ry, band = gc + 0.06, 0.05
    skx, skz = 0.70, 1.72
    rim = []
    for z in (skz, -skz):                                # front + rear skirt bands
        v, f = _box(0.0, ry, z, skx * 2, band, 0.02)
        rim.append(static(v, f, (66, 76, 96)))
    for x in (skx, -skx):                                # side skirt bands
        v, f = _box(x, ry, 0.0, 0.02, band, skz * 2)
        rim.append(static(v, f, (66, 76, 96)))

    def bob(px, pz):                                     # a hanging pendulum weight + gen
        v, f = _box(px, ry - 0.055, pz, 0.028, 0.06, 0.028)
        rim.append(static(v, f, (150, 205, 235)))
        rim.append(_pipe((px, ry - 0.01, pz), (px, ry - 0.03, pz), 0.004, (120, 130, 150)))
    for i in range(6):
        tx = -skx + (i + 0.5) * (2 * skx) / 6.0
        bob(tx, skz); bob(tx, -skz)
    for i in range(9):
        tz = -skz + (i + 0.5) * (2 * skz) / 9.0
        bob(skx, tz); bob(-skx, tz)
    parts.append(Part("pendulum", "Inertial Pendulum Rim (bumper-skirt harvester)", rim,
        ["Function: a shallow ~2-5 in SKIRT around the base of the body (the lower",
         "bumpers, front/back and side to side) that hangs %d small PENDULUM weights." % PENDULUM["weights"],
         "When the car accelerates, brakes, corners, or its pitch changes on a grade,",
         "the effective-gravity vector tips and each weight SWINGS, driving a tiny",
         "generator -- a regenerative damper on the body's own rocking motion.",
         "Up to ~%.0f W in stop-go city, cornering and rolling/incline terrain; ~zero" % PENDULUM["max_w"],
         "on a dead-straight steady cruise. The skirt adds a little drag, so it",
         "DEPLOYS at low speed and RETRACTS flush by ~%.0f mph to avoid a highway drag" % PENDULUM["retract_mph"],
         "penalty -- net MPG gain only where it helps (info 6i)."],
        order=nextord(), explode=(0.0, -1.6, 0.0), color=(150, 205, 235)))

    # --- Exterior lights (headlights + rear light bar + reverse lights) ---
    lights = []
    for sx in (0.24, -0.24):                              # front LED headlights
        v, f = _box(sx, 0.40, 1.80, 0.14, 0.09, 0.05)
        lights.append(static(v, f, (245, 240, 205)))
    v, f = _box(0.0, 0.43, -1.85, 0.46, 0.06, 0.04)       # full-width rear light bar
    bar = static(v, f, (150, 40, 35))                     # dim red; brightens on brake
    bar.brake = True
    lights.append(bar)
    for sx in (0.30, -0.30):                              # reverse lights flanking the bar
        v, f = _box(sx, 0.40, -1.86, 0.09, 0.05, 0.03)
        rl = static(v, f, (120, 120, 108))               # dim; brightens white in reverse
        rl.reverse = True
        lights.append(rl)
    parts.append(Part("lights", "Head + Tail Lights", lights,
        ["Function: low-draw LED headlights, a full-width rear light bar and reverse",
         "lights -- the road-legal exterior lighting on the teardrop shell. The rear",
         "bar brightens on braking and the reverse lights on reverse (see DRIVE)."],
        order=nextord(), explode=(0.0, 1.0, 1.6), color=(245, 240, 205)))

    return parts


class EngineRenderer:
    """Projects + paints the spec'd engine Parts. Supports full / exploded /
    assembly views with an optional cross-SECTION CUT toggle, mouse hover-
    picking, highlight + hover-pop, and the assembly 'puzzle'."""

    def __init__(self, parts_builder=None, supports_pistons=True,
                 home_az=0.65, home_el=0.50, home_dist=1.55):
        self.parts_builder = parts_builder or build_engine_parts
        self.supports_pistons = supports_pistons
        self.n_pistons = 1
        self.parts = (self.parts_builder(self.n_pistons) if supports_pistons
                      else self.parts_builder())
        self._home = (home_az, home_el, home_dist)
        self.az = home_az
        self.el = home_el
        self.dist = home_dist
        self.pan = np.array([0.0, 0.0])
        self.light = np.array([0.4, 0.6, 1.0])
        self.light = self.light / np.linalg.norm(self.light)
        self.view = "full"                  # full | exploded | assembly
        self.section = False                # cross-section CUT toggle (any view)
        self.pedal_extend = 0.0             # pop-out pedal deploy 0..1 (else retracted)
        self.explode_amt = 0.0              # animated 0..1
        self.assembled = len(self.parts)    # parts placed in assembly mode
        self.hovered = None
        self.selected = None
        self.pop = np.zeros(len(self.parts))
        self.hover_spread = 0.0

    def set_pistons(self, n):
        if not self.supports_pistons:
            return
        n = max(1, min(10, n))
        if n == self.n_pistons:
            return
        self.n_pistons = n
        self.parts = self.parts_builder(n)
        self.pop = np.zeros(len(self.parts))
        self.hovered = None
        self.selected = None
        self.assembled = len(self.parts)

    def reset_view(self):
        self.az, self.el, self.dist = self._home
        self.pan = np.array([0.0, 0.0])

    def zoom_at(self, factor, mouse_pos=None, rect=None):
        """Zoom the preview camera around the cursor, with enough range for
        close mechanical inspection and full-assembly framing."""
        old = self.dist
        self.dist = max(0.28, min(9.0, self.dist * factor))
        if old <= 1e-6 or mouse_pos is None or rect is None:
            return
        if not rect.collidepoint(mouse_pos):
            return
        anchor = np.array([mouse_pos[0] - (rect.x + rect.w / 2.0),
                           mouse_pos[1] - (rect.y + rect.h / 2.0)], dtype=float)
        scale = old / self.dist
        self.pan = anchor - (anchor - self.pan) * scale

    def orbit(self, dx, dy, fine=False):
        sens = 0.004 if fine else 0.009
        self.az += dx * sens
        self.el += dy * sens
        self.el = max(-1.55, min(1.55, self.el))

    def pan_by(self, dx, dy, fine=False):
        sens = 0.45 if fine else 1.0
        self.pan += np.array([dx * sens, dy * sens])

    def set_view(self, mode):
        self.view = mode
        if mode == "assembly" and self.assembled >= len(self.parts):
            self.assembled = 0
        self.selected = None

    def toggle_section(self):
        """Cross-section CUT is now a toggle (not its own view): it can be laid
        over FULL or EXPLODED to expose the internal combustion chambers, shaft,
        clutch, flywheel, generator and transmission rings."""
        self.section = not self.section

    def assembly_next(self):
        self.assembled = min(len(self.parts), self.assembled + 1)

    def assembly_prev(self):
        self.assembled = max(0, self.assembled - 1)

    def assembly_all(self):
        self.assembled = len(self.parts)

    def assembly_clear(self):
        self.assembled = 0

    def active_part(self):
        i = self.selected if self.selected is not None else self.hovered
        return self.parts[i] if i is not None else None

    def placing_part(self):
        for p in self.parts:
            if p.order == self.assembled:
                return p
        return None

    def tick(self, dt):
        if self.view != "assembly":
            target = 1.0 if self.view == "exploded" else 0.0
            self.explode_amt += (target - self.explode_amt) * min(1.0, dt * 4)
        hi = self.selected if self.selected is not None else self.hovered
        # hovering in FULL view gently spreads the whole engine open so the
        # focused part is clearly visible (the hovered part pops out further)
        sp_target = 0.30 if (hi is not None and self.view == "full") else 0.0
        self.hover_spread += (sp_target - self.hover_spread) * min(1.0, dt * 5)
        for i in range(len(self.parts)):
            tp = 1.0 if i == hi else 0.0
            self.pop[i] += (tp - self.pop[i]) * min(1.0, dt * 8)

    def _layout(self, pi, vw, eamt):
        part = self.parts[pi]
        if vw == "assembly":
            if part.order < self.assembled:
                return part.explode * 0.0, 1.0, "normal"
            if part.order == self.assembled:
                return part.explode * 0.55, 1.0, "active"
            return part.explode * 1.0, 0.30, "pending"
        return part.explode * eamt, 1.0, "normal"

    def render(self, surf, rect, angles, firing_glow, mouse_pos=None,
               show_labels=True, label_font=None, interactive=False,
               view_override=None, injection=None, heat=0.0,
               selector_radius=None, brake_glow=0.0, reverse_glow=0.0):
        clip = surf.get_clip()
        surf.set_clip(rect)
        cx = rect.x + rect.w / 2.0 + self.pan[0]
        cy = rect.y + rect.h / 2.0 + self.pan[1]
        focal = min(rect.w, rect.h) * 1.12
        Rcam = rot_x(self.el) @ rot_y(self.az)
        default_ang = angles.get("default", 0.0)

        vw = view_override or self.view
        eamt = 0.0 if view_override == "full" else self.explode_amt
        if view_override is None and self.view == "full":
            eamt += self.hover_spread        # open up the engine on hover
        interactive = interactive and view_override is None
        # crisp facet outlines in the main preview at <=2 pistons; off for the
        # small drive inset and heavy 3-4 piston configs (keeps it fast)
        section = (self.section and view_override is None and vw in ("full", "exploded"))
        do_outline = (view_override is None) and (self.n_pistons <= 3 or section)
        hi = (self.selected if self.selected is not None else self.hovered)
        if view_override is not None:
            hi = None

        polys = []
        labels = []
        leaders = []
        screeninfo = []
        fire_points = []            # fixed ignition areas (one per rotary piston)
        lx, ly, lz = float(self.light[0]), float(self.light[1]), float(self.light[2])

        for pi, part in enumerate(self.parts):
            base_off, dim, tag = self._layout(pi, vw, eamt)
            pop = self.pop[pi] if view_override is None else 0.0
            off = base_off + part.popdir * (pop * 0.16)
            if part.key == "pedals":
                # retract the pop-out pedals into the frame (-X) when not deployed
                off = off + np.array([-1.25 * (1.0 - self.pedal_extend), 0.0, 0.0])
            highlight = (pi == hi)
            allcam = []
            for m in part.meshes:
                wv = m.world_verts(angles.get(m.group, default_ang),
                                   selector_radius=selector_radius) + off
                cam = wv @ Rcam.T
                cam[:, 2] += self.dist
                allcam.append(cam)
                caml = cam.tolist()
                col = m.color
                if m.chamber_index is not None:
                    g = firing_glow[m.chamber_index]
                    if g > 0.01:
                        col = _mix(C_CHAMBER, C_CHAMBER_FIRE, g)
                if m.hot and heat > 0.01:
                    col = _mix(col, (255, 70, 30), min(0.72, heat * 0.6))
                if m.brake and brake_glow > 0.01:
                    col = _mix(col, (255, 40, 30), min(0.92, brake_glow))
                if m.reverse and reverse_glow > 0.01:
                    col = _mix(col, (255, 255, 245), min(0.92, reverse_glow))
                if dim < 0.99:
                    col = (int(col[0] * dim), int(col[1] * dim), int(col[2] * dim))
                if highlight:
                    col = _mix(col, (255, 255, 255), 0.28)
                cr, cg, cb = col
                if highlight:
                    outline, ow = C_ACCENT, 2
                elif tag == "active":
                    outline, ow = (255, 210, 120), 2
                elif do_outline:
                    outline, ow = (12, 14, 18), 1
                else:
                    outline, ow = None, 0        # no per-face stroke -> faster
                # project all verts to screen (pure-python is faster per-face)
                sxl, syl, dzl = [], [], []
                for vx2, vy2, vz2 in caml:
                    dzl.append(vz2)
                    if vz2 > 0.05:
                        sxl.append(cx + focal * vx2 / vz2)
                        syl.append(cy - focal * vy2 / vz2)
                    else:
                        sxl.append(0.0)
                        syl.append(0.0)
                # per-mesh DETAIL callouts clutter the assembled FULL view, so show
                # them only when the part is spread out (exploded/assembly) or hovered
                if (show_labels and label_font and m.name and tag != "pending"
                        and (vw != "full" or highlight)):
                    mcen = cam.mean(axis=0)
                    if mcen[2] > 0.05:
                        mlx = cx + focal * mcen[0] / mcen[2]
                        mly = cy - focal * mcen[1] / mcen[2]
                        labels.append((mcen[2], (mlx, mly), m.name, "detail"))
                for face in m.faces:
                    if section and self._section_cut(part.key, wv, face):
                        continue
                    clipped = False
                    for i in face:
                        if dzl[i] <= 0.05:
                            clipped = True
                            break
                    if clipped:
                        continue
                    ax, ay, az = caml[face[0]]
                    bx, by, bz = caml[face[1]]
                    fx, fy, fz = caml[face[2]]
                    ux, uy, uz = bx - ax, by - ay, bz - az
                    wx, wy, wz = fx - ax, fy - ay, fz - az
                    nx = uy * wz - uz * wy
                    ny = uz * wx - ux * wz
                    nz = ux * wy - uy * wx
                    inv = 1.0 / ((nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0)
                    nx *= inv
                    ny *= inv
                    nz *= inv
                    if nz > 0:
                        nx, ny, nz = -nx, -ny, -nz
                    d = nx * lx + ny * ly + nz * lz
                    shade = 0.35 + 0.65 * (d if d > 0.0 else 0.0)
                    fc = (int(cr * shade), int(cg * shade), int(cb * shade))
                    ds = 0.0
                    for i in face:
                        ds += dzl[i]
                    polys.append((ds / len(face), [(sxl[i], syl[i]) for i in face],
                                  fc, outline, ow))

            if not allcam:
                continue
            cam_all = np.vstack(allcam)
            cen = cam_all.mean(axis=0)
            if cen[2] > 0.05:
                safez = np.where(cam_all[:, 2] <= 0.05, 1e9, cam_all[:, 2])
                scx = cx + focal * cam_all[:, 0] / safez
                scy = cy - focal * cam_all[:, 1] / safez
                pcx = cx + focal * cen[0] / cen[2]
                pcy = cy - focal * cen[1] / cen[2]
                rad = float(np.max(np.hypot(scx - pcx, scy - pcy))) * 0.55 + 6
                screeninfo.append((pi, pcx, pcy, rad, cen[2], tag))
                anchor = getattr(part, "fire_anchor", None)
                if anchor is not None and tag != "pending":
                    ca = (anchor + off) @ Rcam.T
                    caz = float(ca[2]) + self.dist
                    if caz > 0.05:
                        fpx = cx + focal * float(ca[0]) / caz
                        fpy = cy - focal * float(ca[1]) / caz
                        fire_points.append((caz, fpx, fpy, max(8.0, rad * 0.30)))
                # PART-name labels follow the L toggle in every view (L off = clean
                # hero view + hover-to-inspect; L on = all part names). The denser
                # per-mesh detail callouts above stay gated to exploded/hover.
                if show_labels and label_font and tag != "pending":
                    labels.append((cen[2], (pcx, pcy), part.name, tag))
                if tag == "active":
                    hc = cen - (off @ Rcam.T)
                    if hc[2] > 0.05:
                        hx = cx + focal * hc[0] / hc[2]
                        hy = cy - focal * hc[1] / hc[2]
                        leaders.append(((pcx, pcy), (hx, hy)))

        polys.sort(key=lambda t: t[0], reverse=True)
        for _, pts, fc, outline, ow in polys:
            if len(pts) >= 3:
                try:
                    pygame.draw.polygon(surf, fc, pts)
                    if outline is not None:
                        pygame.draw.polygon(surf, outline, pts, ow)
                except Exception:
                    pass

        # Fuel-injection + combustion flash at each piston's ONE FIXED ignition
        # area (the rim-top injector). The flash does not travel around the ring;
        # the chamber currently passing that fixed spot is the one that fires.
        # (far-to-near so nearer pistons sit on top)
        glow = float(np.max(firing_glow)) if len(firing_glow) else 0.0
        inj = float(np.max(injection)) if injection is not None and len(injection) else 0.0
        fire_points.sort(key=lambda t: t[0], reverse=True)
        for _, fpx, fpy, crad in fire_points:
            if inj > 0.05:
                draw_injection(surf, fpx, fpy, crad, inj)
            if glow > 0.05:
                draw_combustion(surf, fpx, fpy, crad, glow)

        for a, b in leaders:
            pygame.draw.line(surf, (255, 210, 120), a, b, 1)
            pygame.draw.circle(surf, (255, 210, 120), (int(b[0]), int(b[1])), 5, 1)

        if show_labels and label_font:
            labels.sort(key=lambda t: t[0])
            used = []
            for _, (lx, ly), text, tag in labels:
                ly2 = ly
                for uy in used:
                    if abs(ly2 - uy) < 16:
                        ly2 = uy + 16
                used.append(ly2)
                _label(surf, label_font, text, (lx, ly2),
                       accent=(tag == "active" or tag == "detail"))

        if interactive and mouse_pos is not None:
            mxp, myp = mouse_pos
            best, bestd = None, 1e18
            for pi, pcx, pcy, rad, depth, tag in screeninfo:
                if tag == "pending":
                    continue
                if math.hypot(mxp - pcx, myp - pcy) <= rad and depth < bestd:
                    bestd, best = depth, pi
            self.hovered = best

        surf.set_clip(clip)

    def _section_cut(self, key, wv, face):
        """Remove the +X half of the bulky rotating parts -- a clean half-section
        through the vertical plane. The rim-top combustion chamber (at +Y, x~=0)
        is split right down the middle so its concave pocket, the shaft, clutch
        plates, flywheel, generator and transmission rings all read in true
        cross-section."""
        sectionable = (
            key == "shaft" or key == "trans" or key == "gen" or
            key == "flywheel" or key == "coolring" or
            key.startswith("ring") or key.startswith("clutch") or
            # whole-car parts: cut the +X half so the interior (battery, engine,
            # boiler, cabin) reads in true cross-section in the CAR VIEW.
            key in ("body", "canopy", "battery", "engine", "boiler", "supercap")
        )
        if not sectionable:
            return False
        c = wv[list(face)].mean(axis=0)
        return c[0] > 0.004


def _face_normal(cam_pts, face):
    a = cam_pts[face[0]]
    b = cam_pts[face[1]]
    c = cam_pts[face[2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    n = n / nn
    if n[2] > 0:
        n = -n
    return n


def _mix(c1, c2, t):
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


def draw_combustion(surf, x, y, R, intensity):
    """An expanding orange combustion flash with a hot core + spokes."""
    R = int(max(2, R * (0.55 + 0.9 * intensity)))
    g = pygame.Surface((R * 2 + 4, R * 2 + 4), pygame.SRCALPHA)
    c = R + 2
    pygame.draw.circle(g, (255, 140, 30, int(70 * intensity)), (c, c), R)
    pygame.draw.circle(g, (255, 180, 60, int(120 * intensity)), (c, c), int(R * 0.62))
    pygame.draw.circle(g, (255, 240, 190, int(230 * intensity)), (c, c), int(R * 0.30))
    surf.blit(g, (int(x - c), int(y - c)))
    for k in range(6):
        a = k * math.pi / 3 + intensity * 2.0
        ex, ey = x + math.cos(a) * R * 1.25, y + math.sin(a) * R * 1.25
        pygame.draw.line(surf, (255, 170, 60), (x, y), (ex, ey), 1)


def draw_injection(surf, x, y, R, intensity):
    """A small yellow fuel-injection spray dropping in from the rim injector."""
    top = y - R * (1.6 + 1.2 * intensity)
    col = (255, 230, 120)
    for dx in (-R * 0.4, 0, R * 0.4):
        pygame.draw.line(surf, col, (x + dx * 0.3, top + R * 0.4),
                         (x + dx, y - R * 0.2), 1)
    pygame.draw.circle(surf, col, (int(x), int(top)), max(1, int(R * 0.18)))


def _label(surf, font, text, pos, accent=False):
    col = (255, 210, 120) if accent else C_TEXT
    dot = (255, 210, 120) if accent else C_ACCENT
    img = font.render(text, True, col)
    x, y = int(pos[0]) + 6, int(pos[1]) - 6
    bg = pygame.Surface((img.get_width() + 8, img.get_height() + 4), pygame.SRCALPHA)
    bg.fill((10, 14, 20, 190))
    surf.blit(bg, (x - 4, y - 2))
    pygame.draw.circle(surf, dot, (int(pos[0]), int(pos[1])), 3)
    surf.blit(img, (x, y))


# =============================================================================
# SECTION 4 -- PHYSICS / ENERGY SIMULATION
# =============================================================================

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def hydro_cd_factor():
    """Drag multiplier from the SUPER-HYDROPHOBIC + plastron slip skin: a constant
    dry-air skin-friction cut plus an extra cut in humid/wet air (water beads and
    rolls off, so there is no clinging water film to drag). 1.0 = no coating."""
    return (1.0 - HYDRO["dry_frac"]) * (1.0 - HYDRO["humid_frac"] * clamp(HUMIDITY))


def duct_cd_factor(v_mph):
    """Drag multiplier from the THROUGH-BODY FLOW-THROUGH DUCT: front-grill air runs
    straight through the car and fills the rear WAKE, cutting pressure (form) drag.
    Stronger at speed (the wake vacuum grows with speed). 1.0 = no duct effect."""
    reduction = DUCT["base"] + DUCT["speed"] * clamp(v_mph / DUCT["ref_mph"])
    return 1.0 - reduction * DUCT["blend"]


def effective_cd(v_mph):
    """Effective drag coefficient with ACTIVE MORPHING AERO, the SUPER-HYDROPHOBIC
    slip skin and the THROUGH-BODY FLOW DUCT. Above the onset speed the boat-tail /
    diffuser / underbody skirts deploy, cutting Cd toward morph_cd_min_frac by highway
    cruise; the hydrophobic coating shaves skin-friction / water-film drag; and the
    flow-through duct fills the rear wake to cut pressure drag -- so aero power
    (which grows as v^3) is trimmed exactly where it hurts most."""
    t = clamp((v_mph - AERO["morph_onset_mph"])
              / max(1e-6, AERO["morph_full_mph"] - AERO["morph_onset_mph"]))
    morph = 1.0 - (1.0 - AERO["morph_cd_min_frac"]) * t
    return VEH["Cd"] * morph * hydro_cd_factor() * duct_cd_factor(v_mph)


def effective_crr():
    """Effective rolling-resistance coefficient after the PASSIVE PERMANENT-MAGNET
    wheel bearings remove the bearing-friction slice of the rolling drag (no powered
    coils, near-zero mechanical friction). Lower Crr = lower road load at every speed."""
    return VEH["Crr"] * (1.0 - WHEEL_BEARING["roll_cut_frac"])


def road_load_watts(v_mph, n_pass=1):
    """Steady-cruise road power at the wheels (real road-load physics).
    Returns (total_W, aero_W, roll_W). aero grows as v^3, rolling as v^1.
    Uses the MORPHED effective Cd and the magnetic-bearing effective Crr so both the
    active aero and the passive PM wheel bearings lower the cruise load."""
    v = max(0.0, v_mph) * 0.44704
    m = VEH["curb_mass_kg"] + max(0, n_pass - 1) * OCCUPANT_KG
    f_aero = 0.5 * VEH["air_density"] * effective_cd(v_mph) * VEH["frontal_area_m2"] * v * v
    f_roll = effective_crr() * m * VEH["g"]
    return (f_aero + f_roll) * v, f_aero * v, f_roll * v


def solve_electrical(bus_kw):
    """Model the HV electrical bus for a demanded power (kW) -- same physics for drive
    (traction draw, +) and regen (charge, -). Bus current I = P / V_pack (P = V x I);
    the copper cable drops I^2 x R and the inverter drops (1 - eff) x P. Returns the
    live bus state the ECU commands and the drive HUD reads. The inverter + motor
    efficiencies are the components the aggregate 97% drivetrain figure is built from;
    the cable I^2R is the small extra shown explicitly (wiring is honest, not a guess)."""
    p_w = abs(bus_kw) * 1000.0
    amps = p_w / BUS["pack_v_nom"]
    wire_loss = amps * amps * BUS_R_OHM
    inv_loss = p_w * (1.0 - BUS["inverter_eff"])
    return {"amps": amps,
            "volts": BUS["pack_v_nom"],
            "wire_loss_kw": wire_loss / 1000.0,
            "inv_loss_kw": inv_loss / 1000.0,
            "loss_kw": (wire_loss + inv_loss) / 1000.0,
            "over_limit": amps > BUS["bus_a_max"]}


MPG_BASELINE_REF = {
    # Previous optimizer state before the expanded aero/rolling pass. Used only
    # for the M-chart "gain" ratios, so the graph shows what this reengineering
    # pass changed at each plotted MPH.
    "Cd": 0.09,
    "frontal_area_m2": 1.68,
    "Crr": 0.0032,
    "morph_cd_min_frac": 0.82,
    "morph_onset_mph": 22.0,
    "morph_full_mph": 58.0,
    "hydro_dry": 0.035,
    "hydro_humid": 0.055,
    "bearing_roll_cut": 0.12,
    "duct_base": 0.11,
    "duct_speed": 0.07,
    "duct_blend": 0.65,
    "drag_capture_frac": 0.16,
}


def _baseline_effective_cd(v_mph):
    b = MPG_BASELINE_REF
    t = clamp((v_mph - b["morph_onset_mph"])
              / max(1e-6, b["morph_full_mph"] - b["morph_onset_mph"]))
    morph = 1.0 - (1.0 - b["morph_cd_min_frac"]) * t
    hydro = (1.0 - b["hydro_dry"]) * (1.0 - b["hydro_humid"] * clamp(HUMIDITY))
    duct_reduction = b["duct_base"] + b["duct_speed"] * clamp(v_mph / DUCT["ref_mph"])
    duct = 1.0 - duct_reduction * b["duct_blend"]
    return b["Cd"] * morph * hydro * duct


def baseline_road_load_watts(v_mph, n_pass=1):
    v = max(0.0, v_mph) * 0.44704
    b = MPG_BASELINE_REF
    m = VEH["curb_mass_kg"] + max(0, n_pass - 1) * OCCUPANT_KG
    f_aero = 0.5 * VEH["air_density"] * _baseline_effective_cd(v_mph) \
        * b["frontal_area_m2"] * v * v
    f_roll = b["Crr"] * (1.0 - b["bearing_roll_cut"]) * m * VEH["g"]
    return (f_aero + f_roll) * v, f_aero * v, f_roll * v


def estimate_mpg_baseline(v_mph, n_pass=1, pedals=False):
    if v_mph <= 0.0:
        return MPG_MAX_DISPLAY
    p_wheel, aero_w, _ = baseline_road_load_watts(v_mph, n_pass)
    free_w = ambient_harvest_w(v_mph) + MPG_BASELINE_REF["drag_capture_frac"] * aero_w
    if pedals:
        free_w += n_pass * PEDAL_WATTS_PER_SEAT
    deficit = p_wheel - free_w
    if deficit <= 0.0:
        return MPG_MAX_DISPLAY                      # solar-sustained (no fuel burned)
    kwh_per_mile = (deficit / 1000.0) / v_mph
    gal_per_mile = kwh_per_mile / (FUEL_KWH_PER_GAL * FUEL_TO_WHEEL_EFF)
    return min(MPG_MAX_DISPLAY, 1.0 / gal_per_mile)


def mpg_gain_label(v_mph):
    cur = estimate_mpg(v_mph, 1, False)
    base = estimate_mpg_baseline(v_mph, 1, False)
    if cur >= MPG_MAX_DISPLAY:
        return "solar" if base >= MPG_MAX_DISPLAY else "solar-cvr"
    if base >= MPG_MAX_DISPLAY or base <= 1e-9:
        return "--"
    return "%.1fx" % (cur / base)


def ambient_harvest_w(v_mph=0.0):
    """Total fuel-free AMBIENT harvest (watts), energy-honest:
      SOLAR (film + PV roof) is EXTERNAL energy (from the sun), so it is real free
      power, scaled by SUN and bounded by panel area x irradiance x efficiency.
      The TYRE/TRIBO/PIEZO films only RECOVER a bounded slice of the tyre + body
      hysteresis that the car is ALREADY dissipating as rolling loss -- so they
      reclaim spent energy, never create it, and fall to ZERO when parked (no rolling
      loss). Suspension harvest is handled separately (bump dissipation only)."""
    h = AMBIENT_HARVEST_W
    solar = (h["solar"] + h["solar_roof"]) * SUN
    _, _, roll_w = road_load_watts(v_mph, 1)
    motion_recovery = min(h["triboelectric"] + h["tire"] + h["piezo"],
                          HARVEST_ROLL_FRAC * max(0.0, roll_w))
    return solar + motion_recovery


def solar_boiler_kw(boiler_c=None):
    """Concentrated-sunlight THERMAL power delivered to the boiler by the auto-
    adjusting windshield Fresnel/magnifying lens. Free heat from bright sun that the
    steam expander turns into electricity -- even parked or cruising engine-off. It
    is defocused (regulated) as the boiler nears its saturation temperature so it
    never over-pressures the loop."""
    kw = SOLAR_CONCENTRATOR_KW * SUN
    if boiler_c is not None:
        fade = clamp((PHYS["max_temp_c"] - boiler_c) / 40.0)   # defocus near the top
        kw *= fade
    return kw


def suspension_bump_w(v_mph, grade=0.0):
    """EXTRA electromagnetic-damper harvest on a real (bumpy, graded) road, over the
    flat-cruise baseline already in ambient. Bumps and grade shake the dampers
    harder, so it scales with speed x (road roughness + |grade|)."""
    speed_f = clamp(v_mph / 45.0)
    rough = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 260.0))
    grade_f = clamp(abs(grade) / 0.06)
    return SUSP_BUMP_W_MAX * speed_f * (0.5 * rough + 0.5 * grade_f)


def teg_harvest_kw(block_c):
    """Thermoelectric (Seebeck) trickle straight off the hot block -- a second heat
    path that also keeps producing after the engine stops while the block is hot."""
    return min(TEG["cap_kw"], TEG["kw_per_c"] * max(0.0, block_c - THERM["ambient_c"]))


def steering_regen_kw(v_mph, steer):
    """Small regenerative generator on the steering rack, active only while turning
    at speed."""
    return STEER_REGEN_W_MAX / 1000.0 * clamp(abs(steer)) * clamp(v_mph / 30.0)


def drag_capture_kw(v_mph):
    """PASSIVE DRAG-AIR CAPTURE while moving: the directional micro-grooves funnel a
    slice of the REMAINING aero drag air into tiny bladeless turbines. Recovered
    electricity = groove_eff x turbine_eff of the actual aero drag power (which the
    hydrophobic skin has already minimised), so it scales as v^3 and is always a
    fraction of -- never more than -- the drag it harvests."""
    _, aero_w, _ = road_load_watts(v_mph, 1)
    return DRAG_CAPTURE["groove_eff"] * DRAG_CAPTURE["turbine_eff"] * aero_w / 1000.0


def windmill_kw(v_mph):
    """PARKED WINDMILL: at/below park_mph the rear tail fin deploys as a vertical-axis
    wind turbine and trickle-charges from ambient wind. Retracts (returns 0) while
    driving, where it would only be drag. Scales with the ambient WIND strength."""
    if v_mph > WINDMILL["park_mph"]:
        return 0.0
    gust = 0.45 + 0.55 * abs(math.sin(pygame.time.get_ticks() / 800.0))
    return WINDMILL["max_w"] / 1000.0 * clamp(WIND) * gust


def pendulum_harvest_w(long_g, lat_g, pitch_rate):
    """INERTIAL PENDULUM RIM: the hanging bumper-skirt weights swing when the
    effective-gravity vector in the vehicle frame tips. Harvest scales with the size
    of the inertial events -- and is energy-honest: only DECELERATION (braking dive),
    cornering scrub (lat_g) and incline-change pitch dissipate body-motion energy the
    bobs can reclaim. ACCELERATING stores kinetic energy, so harvesting there would just
    steal traction -- long_g > 0 does NOT harvest. ~Zero on a dead-straight steady
    cruise; bounded by max_w so it only recovers a slice of the body's own rocking."""
    decel = max(0.0, -long_g)                       # brake/dive only, not acceleration
    swing = (clamp(decel / PENDULUM["long_ref"])
             + clamp(abs(lat_g) / PENDULUM["lat_ref"])
             + clamp(abs(pitch_rate) / PENDULUM["pitch_ref"]))
    return PENDULUM["max_w"] * clamp(swing / 3.0)


def pendulum_deploy(v_mph):
    """How far the pendulum skirt is DEPLOYED (1) vs RETRACTED flush (0). It hangs
    out at low speed where its aero drag is negligible and the stop-go swings are
    big, and retracts up into the body at high speed where the skirt drag would cost
    more than it harvests -- so the rim never becomes a net drag penalty."""
    return 1.0 - clamp((v_mph - PENDULUM["deploy_mph"])
                       / max(1e-6, PENDULUM["retract_mph"] - PENDULUM["deploy_mph"]))


def estimate_mpg(v_mph, n_pass=1, pedals=False):
    """Real-life fuel-MPG estimate for a STEADY FLAT cruise (no braking/downhill
    regen assumed -- honest). Gasoline only makes electricity, so:
      fuel power = (road power - FREE harvest) / fuel-to-wheel efficiency,
    where free harvest = EXTERNAL solar + the recover-only harvest already bounded to
    the dissipation the car pays, plus the passengers' pedals if engaged. If solar
    covers the whole road load no fuel burns -> SOLAR-SUSTAINED (shown as a capped
    >figure, not "infinite"). Otherwise MPG = 1 / gal-per-mile."""
    if v_mph <= 0.0:
        return MPG_MAX_DISPLAY
    p_wheel, _, _ = road_load_watts(v_mph, n_pass)
    # free power = SOLAR (external) + the recover-only harvest already bounded to the
    # rolling/aero dissipation the car is paying (drag-air capture) + pedals if engaged.
    free_w = ambient_harvest_w(v_mph) + drag_capture_kw(v_mph) * 1000.0
    if pedals:
        free_w += n_pass * PEDAL_WATTS_PER_SEAT
    deficit = p_wheel - free_w
    if deficit <= 0.0:
        return MPG_MAX_DISPLAY                      # solar covers the load -> no fuel
    kwh_per_mile = (deficit / 1000.0) / v_mph      # kW * hours-per-mile
    gal_per_mile = kwh_per_mile / (FUEL_KWH_PER_GAL * FUEL_TO_WHEEL_EFF)
    return min(MPG_MAX_DISPLAY, 1.0 / gal_per_mile)


def pedals_help(v_mph, n_pass=1):
    """True while pedal power is a meaningful share of the road load; below that
    the pop-out pedals RETRACT into the frame (useless at high speed)."""
    p_wheel, _, _ = road_load_watts(v_mph, n_pass)
    if p_wheel <= 1e-6:
        return True
    return (n_pass * PEDAL_WATTS_PER_SEAT) >= PEDAL_HELP_FRAC * p_wheel


def _fmt_mpg(x):
    return ">9999" if x >= MPG_MAX_DISPLAY else "%4.0f" % x


def effective_chamber_volume_m3():
    """Sealed charge pocket volume for one chamber, derived from dimensions."""
    mm3_to_m3 = 1e-9
    raw = (DIMS["chamber_depth_mm"] * DIMS["chamber_width_mm"]
           * DIMS["ring_thickness_mm"]) * mm3_to_m3
    return raw * PHYS["trapped_volume_factor"]


def pressure_ratio(supercharger_engage):
    """Turbo baseline plus slow automatic supercharger and optional in-chamber PSI."""
    super_pr = boost_multiplier(supercharger_engage)
    air_psi_pr = 1.0 + PHYS["air_psi_optional"] * PHYS["air_psi_pressure_ratio"]
    return PHYS["turbo_pressure_ratio"] * super_pr * air_psi_pr


def thermal_derate(temp_c):
    if temp_c <= PHYS["overheat_c"]:
        return 1.0
    if temp_c >= PHYS["shutdown_c"]:
        return 0.0
    span = PHYS["shutdown_c"] - PHYS["overheat_c"]
    return clamp(1.0 - (temp_c - PHYS["overheat_c"]) / max(1e-6, span), 0.0, 1.0)


def solve_engine_physics(k, combustions, temp_c, coolant_flow, hyd_cmd,
                         firing, slip_load, n_pistons=1, boiler_c=None,
                         fluid=DEFAULT_FLUID, fluid_health=1.0):
    """First-order physical state for one frame.

    Uses the rendered dimensions to calculate air mass per firing event with
    the ideal gas law, then fuel mass, shaft power, generator power, hydraulic
    clamp load, slip heat, and the CLOSED-LOOP steam regeneration:
    two-stage exhaust recovery, a cooling-ring/heat-trap jacket, a geometric heat
    SHIELD that funnels block heat into a compact boiler, a chosen working FLUID
    (low-boil / water / winter), and a DUAL-CHAMBER MULTI-STAGE (compound) expander
    that reuses the pressure internally until weak. Pressure-release regulates
    temperature; a last-resort conditioner is the final backstop.
    """
    if boiler_c is None:
        boiler_c = THERM["ambient_c"]
    fl = FLUIDS.get(fluid, FLUIDS[DEFAULT_FLUID])
    rpm = max(0.0, abs(k.get("piston", 0.0)))
    core_rpm = abs(k.get("core", 0.0))
    ep = clamp(k.get("ep", 0.0))
    c1 = clamp(k.get("c1", 0.0))
    bind = clamp(k.get("bind", 0.0))
    super_eng = clamp(k.get("super", 0.0))
    comb_per_rev = combustion_count(combustions)
    load = ep if firing else 0.0
    derate = thermal_derate(temp_c) if firing else 1.0

    pr = pressure_ratio(super_eng)
    pressure_pa = PHYS["ambient_pa"] * pr
    intake_k = PHYS["intake_temp_k"] + 18.0 * max(0.0, pr - 1.0)
    chamber_v = effective_chamber_volume_m3()
    events_s = rpm / 60.0 * comb_per_rev * max(1, n_pistons) * load
    air_kg_event = (pressure_pa * chamber_v / (PHYS["air_r"] * intake_k)
                    * PHYS["volumetric_eff"] * PHYS["seal_eff"])
    air_kg_s = air_kg_event * events_s
    fuel_kg_s = air_kg_s / PHYS["target_afr"] if firing and derate > 0.0 else 0.0
    gross_kw = fuel_kg_s * PHYS["gasoline_lhv_j_kg"] / 1000.0

    shaft_kw = gross_kw * PHYS["indicated_eff"] * PHYS["mechanical_eff"] * derate
    gear_eff = PHYS["gear_mesh_eff"] ** (1.0 + 0.25 * max(0, k.get("gear", 1) - 1))
    bearing_kw = PHYS["bearing_drag_kw"] * (rpm / 3000.0) ** 2
    net_shaft_kw = max(0.0, shaft_kw - bearing_kw)
    gen_kw = net_shaft_kw * PHYS["generator_eff"] * gear_eff

    hyd_bar = PHYS["hydraulic_max_bar"] * clamp(hyd_cmd)
    main_clamp_kn = hyd_bar * 1e5 * PHYS["main_clutch_area_m2"] / 1000.0
    ring_clamp_kn = hyd_bar * 1e5 * PHYS["ring_bind_area_m2"] / 1000.0
    main_torque_nm = main_clamp_kn * 1000.0 * PHYS["friction_mu_wet"] \
        * PHYS["main_clutch_radius_m"]
    ring_torque_nm = ring_clamp_kn * 1000.0 * PHYS["friction_mu_wet"] \
        * PHYS["ring_bind_radius_m"]
    slip_main_rpm = abs(rpm - core_rpm)
    slip_ring_rpm = abs(core_rpm * max(0.0, k.get("gr", 1.0) - 1.0) * (1.0 - bind))
    slip_main_kw = main_torque_nm * (slip_main_rpm * 2.0 * math.pi / 60.0) / 1000.0
    slip_ring_kw = ring_torque_nm * (slip_ring_rpm * 2.0 * math.pi / 60.0) / 1000.0
    slip_heat_kw = min(PHYS["friction_kw_cap"],
                       (slip_main_kw * (1.0 - c1) + slip_ring_kw * (1.0 - bind))
                       * max(0.0, slip_load))
    friction_kw = slip_heat_kw + bearing_kw

    waste_kw = max(0.0, gross_kw - shaft_kw)
    block_heat_kw = waste_kw * PHYS["heat_to_block_frac"] + friction_kw
    exhaust_kw = waste_kw * (1.0 - PHYS["heat_to_block_frac"])
    flow = clamp(coolant_flow)

    # --- Two-stage exhaust energy recovery -------------------------------
    exh_turbo_kw = exhaust_kw * PHYS["exhaust_recovery_frac"]        # 1st turbo
    exh_left_kw = max(0.0, exhaust_kw - exh_turbo_kw)
    exh_stage2_kw = exh_left_kw * THERM["second_stage_frac"]         # 2nd turbine -> elec
    exh_to_boiler_kw = (exh_left_kw - exh_stage2_kw) * THERM["exhaust_to_boiler_frac"]

    # --- Closed-loop steam recovery (the goal is to HEAT, not cool) ------
    # A heat engine can only do WORK from a temperature difference ABOVE the ambient
    # cold-side sink -- NEVER from ambient heat alone (that would be perpetual motion).
    # So the usable ramp is measured from the fluid's ONSET (the higher of its boil
    # point and ambient) up to saturation: a low-boil fluid (ammonia) starts
    # converting just above ambient and harvests low-grade heat that WATER, which must
    # actually boil, cannot -- but at ambient itself every fluid yields zero.
    onset_c = max(THERM["ambient_c"], fl["boil_c"])
    work_span = max(1.0, fl["full_c"] - onset_c)
    work_ramp = clamp((boiler_c - onset_c) / work_span)   # 0 at/below ambient onset
    # Vapour-pressure gauge (display) still tracks the fluid's own boil point.
    steam_span = max(1.0, fl["full_c"] - fl["boil_c"])
    steam_ramp = clamp((boiler_c - fl["boil_c"]) / steam_span)
    steam_bar = fl["max_bar"] * steam_ramp
    # DUAL-CHAMBER, MULTI-STAGE (compound) reuse: the pressure is routed and
    # rerouted through several internal stages until weak, with two chambers
    # alternating so there is no dead stroke -> higher effective conversion.
    compound_gain = 1.0 + (THERM["compound_stages"] - 1) * THERM["stage_gain"]
    recovery_eff = min(THERM["recovery_eff_cap"],
                       fl["gen_eff"] * compound_gain * THERM["dual_chamber_gain"])
    steam_draw_kw = fl["capacity_kw"] * work_ramp      # heat pulled from the boiler as work
    # aging fluid recovers a bit less until it is serviced (fluid_health 0..1). The
    # recovered electricity is ALWAYS a fraction (recovery_eff < 1) of the heat drawn,
    # so it can never exceed the heat actually flowing through the boiler.
    steam_kw = steam_draw_kw * recovery_eff * (0.55 + 0.45 * clamp(fluid_health))
    recover_kw = steam_kw + exh_stage2_kw              # total recovered elec

    # Passive loss is small: the cooling ring + heat shield KEEP heat_trap_frac.
    passive_kw = PHYS["coolant_ua_kw_c"] * max(0.0, temp_c - THERM["ambient_c"]) \
        * (1.0 - THERM["heat_trap_frac"])
    # Last-resort active thermal conditioning, only when actually overheating.
    condition_kw = 0.0
    if temp_c > PHYS["overheat_c"]:
        condition_kw = THERM["condition_ua_kw_c"] * flow * (temp_c - PHYS["overheat_c"])
    # Geometric heat SHIELD directs block heat into the boiler fluid (block ->
    # boiler). When the block cools it keeps feeding the boiler to sustain steam.
    shield_flow_kw = THERM["shield_ua_kw_c"] * (temp_c - boiler_c)

    heat_in_block = block_heat_kw                        # combustion + friction
    heat_in_boiler = shield_flow_kw + exh_to_boiler_kw   # directed + exhaust heat
    cool_kw = shield_flow_kw + passive_kw + condition_kw  # total heat leaving BLOCK

    status = "NORMAL"
    if temp_c >= PHYS["shutdown_c"]:
        status = "SHUTDOWN"
    elif temp_c >= PHYS["overheat_c"]:
        status = "OVERHEAT"

    return {
        "pressure_kpa": pressure_pa / 1000.0,
        "pressure_ratio": pr,
        "chamber_cc": chamber_v * 1e6,
        "events_s": events_s,
        "air_g_s": air_kg_s * 1000.0,
        "fuel_g_s": fuel_kg_s * 1000.0,
        "afr": PHYS["target_afr"],
        "gross_kw": gross_kw,
        "shaft_kw": shaft_kw,
        "gen_kw": gen_kw,
        "waste_kw": waste_kw,
        "block_heat_kw": block_heat_kw,
        "exhaust_kw": exhaust_kw,
        "friction_kw": friction_kw,
        "cool_kw": cool_kw,
        "orc_kw": recover_kw,
        "heat_in_block": heat_in_block,
        "exh_turbo_kw": exh_turbo_kw,
        "exh_stage2_kw": exh_stage2_kw,
        "exh_to_boiler_kw": exh_to_boiler_kw,
        "steam_kw": steam_kw,
        "steam_bar": steam_bar,
        "steam_max_bar": fl["max_bar"],
        "steam_ramp": steam_ramp,
        "steam_draw_kw": steam_draw_kw,
        "recovery_eff": recovery_eff,
        "fluid": fluid,
        "boiler_c": boiler_c,
        "heat_in_boiler": heat_in_boiler,
        "shield_flow_kw": shield_flow_kw,
        "passive_kw": passive_kw,
        "condition_kw": condition_kw,
        "hyd_bar": hyd_bar,
        "main_clamp_kn": main_clamp_kn,
        "ring_clamp_kn": ring_clamp_kn,
        "bearing_kw": bearing_kw,
        "derate": derate,
        "overheat": temp_c >= PHYS["overheat_c"],
        "shutdown": temp_c >= PHYS["shutdown_c"],
        "status": status,
        "gear_eff": gear_eff,
    }


class Powertrain:
    """Series-hybrid energy economy: rotary generator -> supercap -> battery,
    plus the free-floating flywheel for momentum harvest. Tracks fuel + MPG."""

    def __init__(self):
        self.batt_kwh = ELEC["batt_kwh"]
        self.soc = ELEC["soc_start"]
        self.cap_max = ELEC["supercap_kwh"]
        self.cap_kwh = 0.5 * self.cap_max
        r = DIMS["flywheel_d_mm"] * 0.001 / 2
        self.fly_I = DIMS["flywheel_mass_kg"] * r * r       # rim ~ m r^2
        self.fly_omega = 0.0                                # rad/s
        self.engine_on = False
        self.force_timer = 0.0      # forced engine burst (E key) countdown
        self.block_temp_c = THERM["ambient_c"]   # fed from App for the heat cutoff
        self.ring_rpm = 0.0
        self.clutch1_engage = 0.0
        self.trans_bind = 0.0
        self.supercharger_engage = 0.0
        self.fuel_gal = 0.0
        self.miles = 0.0
        self.engine_seconds = 0.0
        self.total_seconds = 0.0
        self.firing_phase = 0.0
        self.firing_glow = np.zeros(DIMS["chambers"])
        self.regen_active = False
        self.last_mpg = 0.0
        self.soc_target = ENGINE_OFF_SOC             # live predictive charge target
        self._route_net_kw = 0.0                     # deferred traction/regen for 1-pass routing
        self.flow = {"engine": 0.0, "regen": 0.0, "trac": 0.0, "fly": 0.0,
                     "boost": 1.0}

    @property
    def fly_rpm(self):
        return self.fly_omega * 60.0 / (2 * math.pi)

    @property
    def fly_kwh(self):
        return 0.5 * self.fly_I * self.fly_omega ** 2 / 3.6e6

    @property
    def fly_kwh_max(self):
        w = FLYWHEEL_MAX_RPM * 2 * math.pi / 60.0
        return 0.5 * self.fly_I * w * w / 3.6e6

    def _mech(self, dt, firing, moving):
        """Animate the mechanical drivetrain state (ring RPM + clutch/ring bind)."""
        target_rpm = (FIRING_GEN_RPM if firing
                      else (FIRING_IDLE_RPM if moving else 0.0))
        self.ring_rpm += (target_rpm - self.ring_rpm) * min(1.0, dt * 2.5)
        tgt_c1 = 1.0 if firing else (0.25 if moving else 0.0)
        self.clutch1_engage += (tgt_c1 - self.clutch1_engage) * min(1.0, dt * 3)
        self.trans_bind += ((0.9 if firing else 0.15)
                            - self.trans_bind) * min(1.0, dt * 3)

    def route_storage(self, dE, dt):
        """Route signed kWh through the storage buffers with REAL round-trip losses.
        Charging fills the highest-efficiency buffer first -- SUPERCAP, then BATTERY,
        then FLYWHEEL -- and the round-trip loss is taken ON THE WAY IN, so stored
        energy is always LESS than what went in (no free storage). Discharging pulls
        from the fastest buffer first with no extra loss (the loss was paid on charge).
        """
        fly_dE = 0.0
        if dE >= 0.0:
            # CHARGING: supercap (highest RT) -> battery -> flywheel
            into_cap = min(dE, max(0.0, self.cap_max - self.cap_kwh))
            self.cap_kwh += into_cap * STORAGE["supercap_rt"]
            spill = dE - into_cap
            if spill > 0.0:
                room = max(0.0, (ELEC["soc_max"] - self.soc) * self.batt_kwh)
                into_batt = min(spill, room)
                self.soc += into_batt * STORAGE["battery_rt"] / self.batt_kwh
                fly_dE = (spill - into_batt) * STORAGE["flywheel_rt"]   # rest -> flywheel
        else:
            # DISCHARGING: supercap -> battery -> flywheel (loss already paid on charge)
            need = -dE
            from_cap = min(need, self.cap_kwh)
            self.cap_kwh -= from_cap
            need -= from_cap
            if need > 0.0:
                avail = max(0.0, self.soc * self.batt_kwh)
                from_batt = min(need, avail)
                self.soc -= from_batt / self.batt_kwh
                fly_dE = -(need - from_batt)                            # rest <- flywheel

        if fly_dE != 0.0:
            step_h = max(1e-6, dt / 3600.0)
            self.flow["fly"] = fly_dE / step_h
            E = max(0.0, self.fly_kwh + fly_dE) * 3.6e6
            self.fly_omega = math.sqrt(max(0.0, 2 * E / self.fly_I))
            self.fly_omega = min(self.fly_omega,
                                 FLYWHEEL_MAX_RPM * 2 * math.pi / 60.0)
        else:
            self.flow["fly"] *= 0.9

    def update_demo(self, dt):
        """PREVIEW mode: always-firing showcase. Spins the ring and winds the
        flywheel up for display without touching the drive economy (fuel/miles)."""
        self.engine_on = True
        self._mech(dt, True, True)
        target = 9000.0 * 2 * math.pi / 60.0     # gentle visual wind-up
        self.fly_omega += (target - self.fly_omega) * min(1.0, dt * 0.15)

    def update(self, dt, v_mph, road_power_kw, moving, combustions=8, off_soc=None):
        """road_power_kw: + = power demanded at the wheels, - = power available
        from braking / downhill (regen). off_soc is the PREDICTIVE charge target
        set from the look-ahead grade (dynamic SOC window)."""
        self.total_seconds += dt
        self.miles += v_mph * (dt / 3600.0)

        # dynamic SOC window: the look-ahead controller can raise the charge target
        # before a climb (pre-charge) or drop it before a descent (regen headroom).
        off_target = ENGINE_OFF_SOC if off_soc is None else off_soc
        self.soc_target = off_target
        # fire earlier when a target above the resting window is requested (pre-charge)
        on_soc = max(ENGINE_ON_SOC, off_target - 0.30)

        # engine burst controller: fire only when buffers are low, and STOP as soon
        # as EITHER the battery target OR the heat target is reached -- once the
        # block is hot the trickle steam-regen keeps charging with the engine OFF.
        if self.force_timer > 0.0:
            self.force_timer = max(0.0, self.force_timer - dt)
            self.engine_on = True
        elif not self.engine_on and self.soc < on_soc \
                and (self.block_temp_c < ENGINE_OFF_TEMP_C or self.soc < ENGINE_CRIT_SOC):
            self.engine_on = True
        elif self.engine_on and (self.soc >= off_target
                                 or self.block_temp_c >= ENGINE_OFF_TEMP_C):
            self.engine_on = False

        # rotary ring RPM + clutch engagement
        self._mech(dt, self.engine_on, moving)
        target_sc = 1.0 if self.engine_on else 0.0
        rate = SUPERCHARGER_RAMP_RATE if self.engine_on else SUPERCHARGER_RAMP_RATE * 1.6
        self.supercharger_engage += (target_sc - self.supercharger_engage) * min(1.0, dt * rate)

        # engine electrical production + fuel burn
        boost = boost_multiplier(self.supercharger_engage)
        # Actual generator output and fuel burn are applied after the measured
        # kinematics are known by solve_engine_physics().
        engine_kw = 0.0
        if self.engine_on:
            self.engine_seconds += dt
        self.flow["engine"] = engine_kw
        self.flow["boost"] = boost

        trac_kw = max(0.0, road_power_kw) / VEH["drivetrain_eff"]
        regen_kw = max(0.0, -road_power_kw) * VEH["regen_frac"]
        self.flow["trac"] = trac_kw
        self.flow["regen"] = regen_kw

        # DEFER routing: the traction DRAIN is netted against ALL the charge sources
        # (generator + steam + regen + every harvester) in ONE route_storage() call in
        # App.update -- so harvest/generator power used IMMEDIATELY for traction flows
        # straight to the wheel rails and NEVER pays a storage round-trip. Only the true
        # surplus is stored (round-trip loss) or the true deficit pulled. This is a
        # pure routing-synergy win: nothing that is used the same instant is buffered.
        self._route_net_kw = engine_kw + regen_kw - trac_kw

        # flywheel idles down very slowly (magnetic bearings, <0.3% drag)
        self.fly_omega *= max(0.0, 1.0 - dt * 0.0006)
        self.soc = min(ELEC["soc_max"] + 0.001, max(0.02, self.soc))

        if self.fuel_gal > 1e-6:
            self.last_mpg = self.miles / self.fuel_gal
        else:
            self.last_mpg = 9999.0


# =============================================================================
# SECTION 5 -- DRIVE WORLD (procedural hilly route)
# =============================================================================

class DriveWorld:
    REVERSE_MAX = -8.0          # m/s reverse speed cap (~18 mph)

    def __init__(self):
        self.v = 0.0            # m/s, signed (negative = reversing)
        self.dist_m = 0.0
        self.cruise = False
        self.cruise_set = 0.0
        self.steer = 0.0
        self.regen_brake = False
        self.gear = 1           # +1 forward (DRIVE), -1 reverse (REVERSE)
        self.cd_eff = VEH["Cd"]          # live morphed drag coefficient
        self.future_grade = 0.0          # predictive look-ahead grade (radians)
        self.pred_off_soc = ENGINE_OFF_SOC  # predictive SOC charge target

    def grade_at(self, s):
        g = (0.045 * math.sin(s / 220.0)
             + 0.030 * math.sin(s / 70.0 + 1.3)
             + 0.018 * math.sin(s / 33.0 + 2.1))
        return math.atan(g)

    def lookahead_grade(self, ahead_m=PREDICT_LOOKAHEAD_M, samples=5):
        """Average route grade over the next `ahead_m` in the current travel
        direction -- the predictive controller's view of what is coming."""
        direction = 1.0 if self.gear >= 0 else -1.0
        tot = 0.0
        for i in range(1, samples + 1):
            tot += self.grade_at(self.dist_m + direction * ahead_m * i / samples)
        return tot / samples

    def update(self, dt, throttle, brake, pt, force_grade=0.0, combustions=8):
        m = VEH["curb_mass_kg"]
        g = VEH["g"]
        grade = self.grade_at(self.dist_m) + force_grade
        sgn = 1.0 if self.v > 0.02 else (-1.0 if self.v < -0.02 else 0.0)
        self.regen_brake = False
        v_mph_now = abs(self.v) * 2.23694

        # ACTIVE MORPHING AERO: effective Cd falls at cruise -> lower aero force.
        # The INERTIAL PENDULUM SKIRT adds a small Cd penalty while DEPLOYED, but it
        # retracts as speed rises, so the penalty fades out before it can hurt at
        # cruise (this is why it retracts -- the drag would cost more than it harvests).
        cd = effective_cd(v_mph_now) + PENDULUM["cd_penalty"] * pendulum_deploy(v_mph_now)
        self.cd_eff = cd
        # resistive forces (signed along the forward direction)
        f_aero = -0.5 * VEH["air_density"] * cd * VEH["frontal_area_m2"] * self.v * abs(self.v)
        f_roll = -sgn * effective_crr() * m * g * math.cos(grade)
        f_grav = -m * g * math.sin(grade)

        # cruise control (forward only): add energy only when needed
        if self.cruise and self.gear > 0:
            err = self.cruise_set - self.v
            throttle = max(0.0, min(1.0, err * 0.6))
            if err < -0.5:
                brake = min(1.0, -err * 0.25)
                self.regen_brake = True

        rail_force = throttle * VEH["max_rail_kw"] * 1000.0 / max(abs(self.v), 2.0)
        rail_force = min(rail_force, VEH["max_rail_kw"] * 1000.0 / 3.0)
        f_rail = rail_force * self.gear            # rail directly rotates the wheel
        f_brake = -sgn * brake * VEH["max_brake_n"]

        self.v += (f_rail + f_brake + f_aero + f_roll + f_grav) / m * dt
        if self.gear < 0:
            self.v = max(self.v, self.REVERSE_MAX)
        # don't let braking drag the car backwards through a standstill
        if brake > 0 and throttle < 0.05 and abs(self.v) < 0.3:
            self.v = 0.0
        self.dist_m += self.v * dt

        demand_w = abs(f_rail) * abs(self.v) if throttle > 0.01 else 0.0
        regen_w = brake * VEH["max_brake_n"] * abs(self.v)
        if grade < 0:
            regen_w += m * g * (-math.sin(grade)) * abs(self.v) * 0.6
        road_power_kw = (demand_w - regen_w) / 1000.0

        v_mph = self.v * 2.23694

        # PREDICTIVE CONTROL: read the grade ahead and reshape the SOC charge target.
        # Climb ahead (future_grade > 0) -> raise the target and PRE-CHARGE so the
        # hill runs on stored energy; descent ahead -> lower the target to leave
        # regen headroom for the whole downhill.
        self.future_grade = self.lookahead_grade()
        shift = PREDICT_SOC_SWING * clamp(self.future_grade / PREDICT_GRADE_REF, -1.0, 1.0)
        self.pred_off_soc = clamp(ENGINE_OFF_SOC + shift,
                                  ELEC["soc_min"] + 0.05, ELEC["soc_max"])

        pt.regen_active = regen_w > 100.0
        pt.update(dt, abs(v_mph), road_power_kw, moving=abs(self.v) > 0.3,
                  combustions=combustions, off_soc=self.pred_off_soc)
        return grade, v_mph


# =============================================================================
# SECTION 6 -- FULL INFORMATIONAL SPECIFICATION
# =============================================================================

INFO_SECTIONS = [
    ("HOHEV-ROTARY GEN 4  --  CONCEPT OVERVIEW", [
        "A series-hybrid EV whose only fuel burner is an 8-chamber CIRCULAR",
        "rotary 'water-wheel' combustion ring. It runs in short LEAN bursts only to",
        "make electricity; the wheels always run electric -- so fuel is barely used.",
        "It harvests momentum on every coast/brake/downhill, recovers waste heat as",
        "electricity through an AMMONIA steam loop, adds ~%.0f W of always-on ambient" % ambient_harvest_w(0),
        "harvest (solar film+ROOF/suspension/TENG/tyre/piezo), a windshield SUN-lens",
        "boiler, active morphing aero + super-HYDROPHOBIC slip skin + a through-body",
        "FLOW DUCT + passive DRAG-AIR capture + a parked WINDMILL, magnetic wheel",
        "bearings, PCM heat banking, thermoelectrics and predictive control, and even",
        "takes free passenger PEDAL power. Stress/high-speed ~738 MPG; favorable slow",
        "routes can be fuel-free under the M-chart assumptions (see 6c-6h, 8b-8d, M).",
    ]),
    ("1. ROTARY PISTON(S) + FIRING MODEL (the circular 'piston')", [
        "Outer dia %.0f mm | inner hollow %.0f mm | thick %.0f mm (very thin)." % (
            DIMS['ring_outer_d_mm'], DIMS['ring_inner_d_mm'], DIMS['ring_thickness_mm']),
        "%d concave paddles every 45 deg, like a water wheel. A FIXED injector at" % DIMS['chambers'],
        "the rim top direct-injects each chamber as it sweeps past; that chamber",
        "then combusts -- fuel, turbo/supercharger density assist, sealed burn, torque.",
        "The combustion area is FIXED at that rim-top injector -- it does not travel",
        "around the wheel. Whichever chamber is passing the injector is the one",
        "that fires, so the flash always sits at the one ignition spot per piston.",
        "Firing rate is selectable from 1 up to 8 combustions per single rev (keys",
        ", and . or drag the COMBUSTIONS/REV bar). So 100 RPM at 1/rev = 100",
        "combustions/min, and 1000 RPM at 8/rev = 8000 combustions/min.",
        "Run 1-10 stacked pistons on the same core and single transmission for",
        "more torque (preview [ ]).",
    ]),
    ("2. MAIN WET CLUTCH -- MULTI-PLATE, INTEGRAL TO THE PISTON", [
        "The main clutch is BUILT INTO the piston, in its inner bore, and it is a",
        "MULTI-PLATE wet clutch (not a single disc). Steel separator plates splined",
        "to the piston's clutch DRUM interleave with friction plates splined to the",
        "central Core hub -- so the piston gives the clutch a real spot to grip.",
        "Hydraulic fluid pressure squeezes the stack (shared pump/accumulator/valves).",
        "FULL engagement = 1:1 lock, grabs FULL piston power (no slip); less pressure",
        "= the plates SLIP (piston + core faces spin at different rates). More plates",
        "= more slip surface = smoother, more controllable slip. One stack per piston,",
        "all engaging the SAME core.",
    ]),
    ("2b. CLOSED-LOOP STEAM REGEN  (heat it, don't cool it)", [
        "A steam 'cooling' loop whose real job is the OPPOSITE of cooling: it KEEPS",
        "heat to make electricity. A slim COOLING RING hugs the piston and absorbs",
        "combustion + clutch-slip heat into a CLOSED-LOOP fluid (~%.0f%% retained)." % (
            THERM["heat_trap_frac"] * 100),
        "A GEOMETRIC HEAT SHIELD -- angled reflector vanes, not a metal mass -- aims",
        "that heat onto one focus of a COMPACT boiler (sized to scale, not oversized).",
        "PRESSURE, not heat, is the currency: a DUAL-CHAMBER valve block (two chambers",
        "alternating so there is no dead stroke) feeds a MULTI-STAGE COMPOUND expander",
        "that routes and REROUTES the pressure stage to stage, reusing it until weak",
        "-- effectively many small steam engines in series (%d internal stages)." % THERM["compound_stages"],
        "We regulate temperature by RELEASING pressure through a vent (which also",
        "makes power); only if still too hot does a last-resort CONDITIONER cool.",
        "WORKING FLUID (key F): the loop does NOT run water. Water needs ~%.0f C and" % FLUIDS["WATER"]["boil_c"],
        "FREEZES AT 0 C -- a winter liability that can crack the loop. The DEFAULT is",
        "AMMONIA: it flashes at ~%.0f C (harvesting low-grade heat water can't), makes" % FLUIDS["AMMONIA"]["boil_c"],
        "the HIGHEST pressure/power AND withstands the most heat (ok to ~%.0f C) over" % FLUIDS["AMMONIA"]["heat_ok_c"],
        "the LONGEST service cycle -- it takes the abuse of a long-duration loop. It",
        "never freezes (%.0f C). R245FA and methanol are non-freezing alternates." % FLUIDS["AMMONIA"]["freeze_c"],
        "Head-to-head every non-freezing fluid beats water (ammonia ~2.2x water's kWh).",
        "FLUID SERVICE: the charge is consumed per kWh recovered (~%.0f kWh for ammonia)," % FLUIDS["AMMONIA"]["service_kwh"],
        "aging slightly faster if run hotter than it tolerates; recovery tapers until",
        "you change it (F) -- a real service interval per miles / kW produced.",
        "Because the block banks the heat, the expander keeps trickle-charging with",
        "the engine OFF -- extending range.",
    ]),
    ("2b2. REGEN WHEELS + 2-STAGE EXHAUST + ENGINE-OFF (max harvest)", [
        "REGEN WHEELS: the same EM rails that turn the rims run as GENERATORS on",
        "braking and downhill, pouring momentum straight into supercaps + battery. This",
        "is the primary momentum harvest; the steam loop is the primary heat harvest.",
        "2-STAGE EXHAUST: exhaust first drives the intake TURBO, then a 2nd-stage",
        "RECOVERY TURBINE catches ~%.0f%% of what is left (a small generator), and" % (
            THERM["second_stage_frac"] * 100),
        "the STILL-remaining exhaust heat is routed into the boiler. Turbo -> 2nd",
        "turbine -> boiler: almost nothing is wasted.",
        "ENGINE-OFF LOGIC: the burner stops the moment EITHER the battery target OR",
        "the heat target (%.0f C) is reached. Once the block is hot, the trickle" % ENGINE_OFF_TEMP_C,
        "steam-regen + wheel regen keep charging with the engine OFF, so we maximize",
        "engine-off time. When both the charge and the stored heat run down, a short",
        "burst tops them back up.",
    ]),
    ("2c. TURBO + SLOW AUTOMATIC SUPERCHARGER", [
        "A compact exhaust-driven turbo sits at the hot rim: exhaust enters the",
        "turbine side, and the compressor returns boosted intake pressure to the",
        "injector side as baseline charge density.",
        "A compact mechanical supercharger is added back as a second stage. It",
        "does not snap on; its automatic clutch ramps in slowly only while the",
        "engine is running, raising effective air density/compression.",
        "Output now varies with solved pressure ratio, air mass, fuel flow,",
        "temperature derate, gear efficiency and generator efficiency.",
    ]),
    ("2d. PRESSURE SEAL + AIR-PSI CHARGE ASSIST", [
        "A normal piston moves up/down to make high compression before each firing.",
        "This circular piston wheel does not; it mainly spins, so the design should",
        "not depend on a long physical compression stroke for every chamber.",
        "Turbo boost and the slow automatic supercharger substitute for much of",
        "that physical compression by increasing air density before firing.",
        "Internal air-PSI injectors are optional. They are not before or after the",
        "combustion event; they are only a median dense-air assist inside the",
        "sealed piston chamber itself.",
        "If enabled, 1/rev assists one chamber; 8/rev assists every chamber in",
        "that rotation.",
        "A stationary seal shoe and floating seal bars close the shaped pocket at",
        "the injector zone so the pressure is held when ignition happens.",
        "The chamber geometry is conformed to catch the explosion and force the",
        "combustion load into rotation instead of leaking around the wheel.",
        "The gate therefore guarantees sealing, swirl, charge density, and timing",
        "for the circular flywheel piston without making it a reciprocating piston.",
    ]),
    ("2e. FIRST-ORDER PHYSICS ENGINE", [
        "The simulator now calculates behavior from real first-order factors, not",
        "only animation curves. It solves sealed chamber volume (%.0f cc), intake" % (
            effective_chamber_volume_m3() * 1e6),
        "pressure ratio, ideal-gas air mass, target AFR %.1f:1, fuel LHV," % PHYS["target_afr"],
        "gross combustion kW, shaft kW, gear-mesh/generator efficiency and fuel",
        "grams per second.",
        "Hydraulic pressure is converted to clamp force on the in-piston wet",
        "clutch and transmission ring bind. Slip speed x wet-friction torque",
        "becomes heat, so bad clutch settings can overheat the engine.",
        "That heat is then TRAPPED (not cooled): the cooling ring keeps ~%.0f%%," % (
            THERM["heat_trap_frac"] * 100),
        "a geometric shield DIRECTS it into a two-node model (block + boiler fluid),",
        "and the boiler steam DRAW (scaled by the fluid's vapour pressure, boosted",
        "by the %d-stage compound + dual-chamber reuse) pulls heat back out --" % THERM["compound_stages"],
        "venting that pressure is the regulation. The steam turbine + 2nd-stage",
        "exhaust turbine recover electricity; above %.0f C the engine derates, and" % PHYS["overheat_c"],
        "by %.0f C it is a thermal shutdown (the last-resort conditioner fights that)." % PHYS["shutdown_c"],
    ]),
    ("3. TUNGSTEN KINETIC FLYWHEEL -- ON THE CORE", [
        "Weighted disc, %.0f mm, %.0f kg tungsten, mounted ON the central core." % (
            DIMS['flywheel_d_mm'], DIMS['flywheel_mass_kg']),
        "When the MAIN CLUTCH disengages the disc FREE-SPINS with its stored",
        "energy -- the transmission can take that energy back, or use it to",
        "reduce stall. Wound up by downhill / coast momentum while the engine is",
        "OFF. Magnetic bearings, <0.3%% drag, safe to %.0f RPM." % FLYWHEEL_MAX_RPM,
    ]),
    ("4. TRANSMISSION = GEAR WET CLUTCH (concentric ring layers)", [
        "%d concentric RING LAYERS -- and each ring IS its own GEAR RATIO. The" % DIMS['trans_rings'],
        "rings act like a second, multi-plate WET CLUTCH just for the gears: they",
        "BIND together to engage. The TRANSMISSION BIND (Trans Wet Clutch) slider",
        "is that wet-clutch pressure -- FULL bind = the selected gear locks solid",
        "(1:1, no slip); LESS bind = MORE SLIP. So the lower the transmission bind,",
        "the more the selected gear (e.g. gear 2) slips against the core.",
        "A separate SET-GEAR shifter (the Gear slider) picks WHICH ring is engaged;",
        "the rest stay disengaged. Every ring carries TEETH on its INNER edge and",
        "the receival gear shifts to the selected ring edge. Lives inside the",
        "hollow (thin engine). Rings, not discs.",
        "Ratios are calculated from actual ring pitch radii: %.2f:1 -> %.2f:1." % (
            trans_ratios()[0], trans_ratios()[-1]),
    ]),
    ("5. OUTPUT GEAR + GENERATOR (charges battery/caps)", [
        "A toothed receival gear MESHES the transmission ring teeth and rotates on",
        "its OWN axis (a normal gear), with a small unavoidable ~%.0f deg pivot" % DIMS['out_angle_deg'],
        "angle loss. It drives the axial-flux generator, whose output CHARGES the",
        "batteries + capacitors -- it does NOT drive the wheels. The battery energizes",
        "the wheel rails directly; this engine is only a range-extender.",
    ]),
    ("6. ELECTRIC + ENERGY STRATEGY", [
        "Battery %.0f kWh held in a %.0f-%.0f%% SOC window for max regen room." % (
            ELEC['batt_kwh'], ELEC['soc_min'] * 100, ELEC['soc_max'] * 100),
        "Supercapacitor buffer %.2f kWh handles real-time charge/drain." % ELEC['supercap_kwh'],
        "Routing priority: engine/regen/pedals -> supercap -> battery -> flywheel.",
        "Engine fires only below %.0f%% SOC, stops at %.0f%% OR once the block is hot" % (
            ENGINE_ON_SOC * 100, ENGINE_OFF_SOC * 100),
        "enough (%.0f C) to keep the steam loop charging on its own. Never chase a" % ENGINE_OFF_TEMP_C,
        "full battery -- drain to the next downhill and harvest it back.",
    ]),
    ("6b. PASSENGER PEDAL-ASSIST (pedal like a bike for range)", [
        "Every seat has a POP-OUT foot-pedal generator. Pushing the pedals (exactly",
        "like pedaling a bike) spins a mini generator that TRICKLE-charges the",
        "battery. The pedals are positionable and pressable forever, so the charge",
        "is cumulative -- passengers can add range indefinitely by pedaling.",
        "~%.0f W per seat x %d seats = up to %.0f W of FREE human power (toggle K)." % (
            PEDAL_WATTS_PER_SEAT, DIMS['seats'], PEDAL_WATTS_PER_SEAT * DIMS['seats']),
        "It is small next to traction, but it is FUEL-FREE, so it directly extends",
        "engine-off time and lifts MPG -- in the sim, pedaling at ~75 mph raised MPG",
        "from ~900 to ~1000+ while cutting engine runtime. Great on long descents,",
        "traffic and cruise where every free watt keeps the burner off longer.",
    ]),
    ("6c. WHY THE MPG IS INSANELY HIGH -- HOW IT IS ACHIEVABLE", [
        "MPG is fuel-only miles / gallons burned. The wheels NEVER burn fuel -- they",
        "run on electricity. Gasoline is used ONLY as a last-resort range-extender to",
        "top the battery, so the trick is simple: make the electricity almost free",
        "and burn fuel almost never. Seven stacked effects do that:",
        "1) REDESIGNED EFFICIENT ENGINE: higher compression + tighter seals + low-",
        "   friction, magnetic-bearing internals -> ~%.1f%% fuel-to-wheel, little loss." % (
            FUEL_TO_WHEEL_EFF * 100),
        "2) RIGHT-SIZED LEAN BURST: corrected sweep picks %.0f RPM at 8 combustions/rev" % FIRING_GEN_RPM,
        "   -- same useful charge at low bearing speed, with bounded heat recovery.",
        "3) HEAT REGEN: the ammonia %d-stage steam loop turns trapped waste heat" % THERM['compound_stages'],
        "   back into electricity, and keeps charging even after the burner stops.",
        "4) MOMENTUM REGEN: wheel rails recover ~%.0f%% of brake + downhill;" % (VEH['regen_frac'] * 100),
        "   descents also wind the free-floating tungsten flywheel via the clutch.",
        "5) KILLED THE DRAG: AI-optimized body Cd %.2f, %.0f kg, so cruising asks only" % (
            VEH['Cd'], VEH['curb_mass_kg']),
        "   a fraction of the power the other harvests cover outright.",
        "6) AMBIENT HARVEST: ~%.0f W of EXTERNAL solar (sun) in good light, plus" % (
            (AMBIENT_HARVEST_W["solar"] + AMBIENT_HARVEST_W["solar_roof"]) * SUN),
        "   tyre/suspension harvest that only RECLAIMS rolling/bump loss (never free).",
        "7) FREE HUMAN WATTS: passenger pedals trickle in more fuel-free charge.",
        "Real-life estimate (M chart): a gentle <=10 mph cruise in sun = SOLAR-",
        "SUSTAINED (no fuel); 50 mph improves %s -> %s MPG (%s), 4 pedalers %s;" % (
            _fmt_mpg(estimate_mpg_baseline(50, 1, False)),
            _fmt_mpg(estimate_mpg(50, 1, False)), mpg_gain_label(50),
            _fmt_mpg(estimate_mpg(50, 4, True))),
        "80 mph = ~%.0f MPG. Stress/high-speed ~738; favorable slow routes can be fuel-free --" % estimate_mpg(80, 1, False),
        "reached by hardly ever burning fuel, not by magic.",
    ]),
    ("6d. REAL-LIFE MPG vs SPEED  (press M for the chart)", [
        "REAL road-load estimates (steady flat cruise, no regen -- honest, not game",
        "numbers). Fuel only makes electricity, so MPG = how little road power is",
        "left after the free pedal watts. Road power = (aero + rolling) x speed, and",
        "AERO POWER GROWS AS SPEED CUBED -- that single fact is the whole story.",
        "",
        "  speed   aero%   no-ped   1ped  2ped  3ped  4ped",
        *["  %2d mph   %2.0f%%    %s    %s  %s  %s  %s" % (
            v, road_load_watts(v, 1)[1] / max(1e-6, road_load_watts(v, 1)[0]) * 100,
            _fmt_mpg(estimate_mpg(v, 1, False)), _fmt_mpg(estimate_mpg(v, 1, True)),
            _fmt_mpg(estimate_mpg(v, 2, True)), _fmt_mpg(estimate_mpg(v, 3, True)),
            _fmt_mpg(estimate_mpg(v, 4, True)))
          for v in MPG_SPEEDS_MPH],
        "",
        "READ IT: at 5-10 mph the road load is only tens-to-hundreds of watts, so",
        "the passengers' pedals (%.0f W each) can cover it ALL -> fuel-free, zero fuel." % PEDAL_WATTS_PER_SEAT,
        "More passengers = more free watts = fuel-free up to a higher speed (their",
        "extra mass barely matters when rolling load is tiny). As speed climbs, aero",
        "takes over (%.0f%% of the load at 5 mph, %.0f%% at 80 mph), the pedals become" % (
            road_load_watts(5, 1)[1] / max(1e-6, road_load_watts(5, 1)[0]) * 100,
            road_load_watts(80, 1)[1] / max(1e-6, road_load_watts(80, 1)[0]) * 100),
        "a rounding error and RETRACT into the frame, and MPG falls with speed^3:",
        "~%.0f MPG at 5 mph collapses to ~%.0f MPG at 80 mph. 80 mph is 'bad' ONLY" % (
            estimate_mpg(5, 1, False), estimate_mpg(80, 1, False)),
        "next to a slow cruise -- it is still a strong ~%.0f MPG, but easing off" % estimate_mpg(80, 1, False),
        "multiplies it 5-15x. Winning strategy: go slower, carry pedalers.",
    ]),
    ("6e. MULTI-LAYER HARVEST (external solar + honest recovery)", [
        "Energy-honest: the ONLY true free input is EXTERNAL solar (from the sun). The",
        "motion harvesters do NOT create energy -- they RECLAIM a bounded slice of loss:",
        " - SOLAR (external): PV roof + quantum-dot film (~%.0f W good sun)." % (
            (AMBIENT_HARVEST_W["solar"] + AMBIENT_HARVEST_W["solar_roof"]) * SUN),
        " - SUSPENSION: EM dampers reclaim BUMP dissipation only (-> 0 on smooth road),",
        "   while actively controlling the ride.",
        " - TRIBOELECTRIC/TYRE/PIEZO: reclaim a slice of the tyre + body HYSTERESIS that",
        "   rolling resistance already dissipates (capped to %.0f%% of rolling loss)." % (
            HARVEST_ROLL_FRAC * 100),
        "So in bright sun, below ~10-25 mph the SOLAR alone can cover the whole road",
        "load -> fuel-free (solar-sustained), no pedals and no fuel.",
        "DOWNHILL FLYWHEEL: on descents, gravity + inertia wind the free-floating",
        "%.0f kg tungsten flywheel through the low-slip clutch (no combustion), and" % DIMS['flywheel_mass_kg'],
        "braking regen recovers ~%.0f%% by back-driving the wheel rails into the supercaps." % (VEH['regen_frac'] * 100),
        "HEADROOM IS KING: the battery is held in a tight %.0f-%.0f%% window so there" % (
            ELEC['soc_min'] * 100, ELEC['soc_max'] * 100),
        "is always empty room to swallow the next brake, hill and harvest. The rotary",
        "fires only when battery + flywheel + harvest cannot meet demand, then stops.",
        "Stress/high-speed ~738 MPG; favorable slow/harvest-rich routes can be fuel-free.",
    ]),
    ("6f. NEW EFFICIENCY STACK  (active aero, PCM, TEG, predictive)", [
        "V1.17 stacks five more fuel-free levers on top of the above -- each shown",
        "LIVE in the DRIVE 'EFFICIENCY STACK' panel on the left:",
        " - ACTIVE MORPHING AERO: a shape-memory boat-tail, rear diffuser and under-",
        "   body skirts deploy at cruise, cutting the effective Cd from %.3f toward" % VEH["Cd"],
        "   %.3f (%.0f%% of static) exactly where aero drag (v^3) hurts most -- the" % (
            effective_cd(80), AERO["morph_cd_min_frac"] * 100),
        "   single biggest cruise-MPG gain. The M chart already reflects it.",
        " - REGEN SUSPENSION (dynamic): the electromagnetic dampers harvest up to",
        "   ~%.0f W EXTRA on rough/graded roads, above the ~%.0f W flat baseline." % (
            SUSP_BUMP_W_MAX, AMBIENT_HARVEST_W["suspension"]),
        " - THERMOELECTRIC (TEG): Seebeck films on the block make up to ~%.1f kW" % TEG["cap_kw"],
        "   straight from block-to-ambient heat -- a 2nd heat path that keeps",
        "   producing after shutdown while the block is still hot.",
        " - PCM HEAT BANK: a phase-change material soaks up boiler heat while firing",
        "   and bleeds it back for minutes after, so the steam loop keeps charging",
        "   with the engine OFF -- visibly stretching engine-off time.",
        " - PREDICTIVE CONTROL + DYNAMIC SOC: the controller reads ~%.0f m of grade" % PREDICT_LOOKAHEAD_M,
        "   ahead and shifts the charge target -- PRE-CHARGE before a climb, and empty",
        "   the buffers before a descent so the whole downhill is regenerated, not",
        "   dumped as brake heat. The burner fires less and catches more free energy.",
        " - REGEN STEERING (~%.0f W turning) + seat/body PIEZO add a final trickle." % STEER_REGEN_W_MAX,
        "Net: demand down, recovery up -- MPG climbs above the old figures.",
    ]),
    ("6g. SOLAR ROOF + WINDSHIELD SUN-CONCENTRATOR (sunshine -> power)", [
        "Two dedicated SOLAR additions harvest the sun two different ways -- both are",
        "real parts in the 3D assembly (hover them) and both raise power + MPG:",
        " - SOLAR ROOF PANEL: a high-efficiency PV roof ABOVE the quantum-dot film",
        "   adds ~%.0f W of FREE electric straight to the pack. With the film that is" % AMBIENT_HARVEST_W["solar_roof"],
        "   ~%.0f W of total solar -- pure traction the burner never has to make." % (
            (AMBIENT_HARVEST_W["solar"] + AMBIENT_HARVEST_W["solar_roof"]) * SUN),
        " - WINDSHIELD SUN CONCENTRATOR: an auto-adjusting Fresnel / magnifying lens",
        "   in the windshield tracks the sun and FOCUSES it onto a receiver on the",
        "   boiler, dumping up to ~%.2f kW of free HEAT into the closed-loop fluid." % SOLAR_CONCENTRATOR_KW,
        "   The steam expander turns that sunshine into ELECTRICITY -- even parked or",
        "   cruising engine-off -- and it defocuses near saturation so it can't over-",
        "   pressure. Bright HOT sun (when you're driving in the heat) = more power.",
        "The roof is a DIRECT-electric path; the lens is a HEAT path through the steam",
        "loop. Because the wheels are electric, both simply mean the engine fires even",
        "less often -- so on a sunny day the MPG numbers climb further still.",
    ]),
    ("6h. PASSIVE DRAG-AIR CAPTURE + PARKED WINDMILL (recover the drag)", [
        "The hydrophobic slip skin (8b) makes MOST airflow slip, but the little that",
        "still grips the body is not wasted -- it is CAUGHT and turned into electricity:",
        " - DRAG-AIR CAPTURE (moving): directional MICRO-GROOVES (deep shark-skin",
        "   riblets aligned WITH the airflow) funnel the remaining boundary-layer air",
        "   through internal ducts into tiny low-drag BLADELESS (Tesla-type) turbines.",
        "   Fully PASSIVE -- no pumps or valves. It recovers ~%.0f%% of the REMAINING" % (
            DRAG_CAPTURE["groove_eff"] * DRAG_CAPTURE["turbine_eff"] * 100),
        "   aero drag power (groove %.0f%% x turbine %.0f%%), scaling with speed (aero" % (
            DRAG_CAPTURE["groove_eff"] * 100, DRAG_CAPTURE["turbine_eff"] * 100),
        "   power ~ v^3) -- so it gives the most on the highway. It always recovers",
        "   LESS than the drag (no free energy); it just stops WASTING it. On a fast",
        "   cruise it adds a few percent of effective efficiency for zero fuel.",
        " - PARKED WINDMILL: when parked (or crawling <= %.0f mph) a rear tail fin" % WINDMILL["park_mph"],
        "   EXTENDS as a small VERTICAL-AXIS wind turbine and trickle-charges from",
        "   ambient WIND (up to ~%.0f W) alongside the solar roof -- free range over" % WINDMILL["max_w"],
        "   hours/days with zero driver input. It RETRACTS while driving (where it",
        "   would only be drag). Deploys passively on an anemometer / shape-memory",
        "   trigger. Watch the DRAG CAPTURE + WIND rows in the drive panel.",
        "Both work WITH the skin (the skin cuts drag first, the grooves catch the",
        "remainder), so nothing is left on the table -- more MPG moving, free charge",
        "parked.",
    ]),
    ("6i. INERTIAL PENDULUM RIM (harvest the body's own motion)", [
        "A shallow ~2-5 in SKIRT rings the base of the body -- the lower bumpers,",
        "front/back and side to side -- and hangs %d small PENDULUM weights." % PENDULUM["weights"],
        "Whenever the car ACCELERATES, BRAKES, CORNERS, or its pitch changes on a",
        "gradient, the effective-gravity vector in the vehicle frame tips, so every",
        "hanging weight SWINGS back and forth. Each swinging weight-frame drives a tiny",
        "generator -- so the body's own inertial rocking (normally wasted as it damps",
        "out) becomes electricity. It is a REGENERATIVE inertial damper.",
        "It gives the most exactly where the other harvesters give least: STOP-GO city",
        "driving, tight cornering, and rolling / incline-changing terrain (up to ~%.0f W)," % PENDULUM["max_w"],
        "and ~ZERO on a dead-straight steady cruise. Bounded by the mass of the bobs,",
        "so it only ever recovers a slice of the rocking -- never more than it carries.",
        "SMART RETRACTION: the dangling skirt adds a little AERO DRAG (Cd +%.3f when" % PENDULUM["cd_penalty"],
        "out), and drag grows with speed^2 -- so above a threshold it would cost more",
        "than it harvests. It therefore DEPLOYS at low speed (drag tiny, stop-go swings",
        "big -> extra MPG) and RETRACTS flush into the body from ~%.0f mph, fully gone" % PENDULUM["deploy_mph"],
        "by ~%.0f mph, so it NEVER becomes a net drag penalty on the highway. Watch the" % PENDULUM["retract_mph"],
        "PENDULUM RIM row flip to RETRACTED and the bobs pull up on the car in DRIVE.",
    ]),
    ("7. PREVIEW DETAIL + SECTION-CUT TOGGLE", [
        "The preview renderer uses higher geometric mesh resolution so circular",
        "parts, housings, ring teeth, shafts and internal bores read more clearly.",
        "The engine is drawn larger on screen but remains dimensionally to scale.",
        "SECTION-CUT is now a TOGGLE (key 4 or X), not its own view -- switch it on",
        "over FULL or EXPLODED. It half-sections the bulky rotating assembly through",
        "the vertical plane, splitting the rim-top combustion chamber right down the",
        "middle so its detailed concave pocket, plus the multi-plate clutch, shaft,",
        "flywheel, generator and transmission rings, all read in true cross-section.",
        "The combustion pocket geometry is built to scale straight from DIMS.",
        "In CAR VIEW the four wheel drives are also explicit: each wheel is its own",
        "hoverable rail-ring module with the in-rim rotor disc, EM stator rail,",
        "air-gap ring, copper winding, pole markers and magnetic-bearing hub drawn.",
    ]),
    ("8. VEHICLE + AI-OPTIMIZED AERO BODY (kill the drag)", [
        "The single biggest real-world MPG lever is AERO DRAG, so the body was shape-",
        "optimized in an AI generative-design + CFD loop to the physical minimum: a",
        "low, narrow TEARDROP cabin with a long BOAT-TAIL, faired/covered wheels and",
        "a fully sealed flat underbody. That drops Cd to ~%.2f (from ~0.13) and the" % VEH['Cd'],
        "frontal area to %.2f m^2 -- roughly HALVING the aero force at any speed." % VEH['frontal_area_m2'],
        "A generative-lattice ultralight structure cuts curb mass to ~%.0f kg and" % VEH['curb_mass_kg'],
        "low-rolling-resistance tyres take Crr to %.4f, so the whole road load is" % VEH['Crr'],
        "tiny. Four EM RAIL-RING direct wheel drives (~%.0f kW, rim-as-rotor, gearless, info 11)" % VEH['max_rail_kw'],
        "with low-loss SiC inverters give ~%.1f%% drivetrain and true zero-drag coasting." % (
            VEH['drivetrain_eff'] * 100),
        "Net: real-life road load at 50 mph fell by ~35%%, so MPG at 50 mph roughly",
        "DOUBLED (see the M chart) -- and slower still is far better again.",
        "The body also wears the SUPER-HYDROPHOBIC slip skin (8b) for even less drag.",
    ]),
    ("8b. SUPER-HYDROPHOBIC + PLASTRON SLIP SKIN (humid/wet-air drag cut)", [
        "The whole body wears a nano-textured SUPER-HYDROPHOBIC coating. Two effects,",
        "both PURE drag reduction -- so both directly raise MPG:",
        " - WATER ROLLS OFF: rain and humidity BEAD UP (very high contact angle) and",
        "   roll straight off instead of forming a clinging water FILM. Ordinary paint",
        "   in wet/humid air drags -- and carries -- a skin of water; this coating",
        "   sheds it, so wet or HUMID air costs far less. The wetter the air, the",
        "   bigger the win: it scales with humidity (now RH %2.0f%%)." % (HUMIDITY * 100),
        " - VACUUM-LIKE AIR SLIP: the same micro-texture RETAINS a thin trapped AIR",
        "   layer (a 'plastron'). The passing airflow then rides on that air cushion",
        "   instead of gripping the paint -- a near-frictionless, VACUUM-LIKE slip",
        "   boundary that cuts turbulent SKIN-FRICTION drag even in dry air (the",
        "   shark-skin riblet effect).",
        "Effect on the effective Cd: ~%.1f%% cut in dry air, up to ~%.1f%% total in" % (
            HYDRO["dry_frac"] * 100, (1.0 - hydro_cd_factor()) * 100),
        "humid air. It folds straight into the Cd ON TOP of the active morphing aero,",
        "so the M chart AND the drive economy both already show the lower drag (watch",
        "the beads roll off the car in DRIVE, and the RH%% + drag-cut in the panel).",
        "It is also self-cleaning and anti-icing. Because aero drag grows as speed",
        "CUBED and is the single biggest real-world MPG lever, shaving even a few",
        "percent of Cd for free -- and much more in the rain -- lifts the mileage.",
    ]),
    ("8c. PASSIVE PERMANENT-MAGNET WHEEL BEARINGS (kill parasitic drag)", [
        "The wheels do NOT ride on conventional ball/taper bearings (rolling friction),",
        "NOR on active ELECTROMAGNETIC bearings -- those burn continuous power in their",
        "control coils, a parasitic electrical load. They ride on HARD-MAGNETIC, PASSIVE",
        "permanent-magnet levitation: a Halbach magnet array floats each wheel hub on",
        "repelling fields with essentially ZERO mechanical friction and ZERO power draw.",
        "Permanent-magnet levitation is statically unstable on its own (Earnshaw's",
        "theorem), so it is STABILIZED PASSIVELY -- a diamagnetic pyrolytic-graphite",
        "element, a low-drag ceramic touchdown race and an eddy-current damper pin the",
        "last axis without any powered coils -- so it stays stable AND draws no energy.",
        "Effect: the bearing-friction slice of the rolling/parasitic drag is nearly",
        "gone -- effective Crr %.4f -> %.4f (~%.0f%% less), lowering the road load at" % (
            VEH["Crr"], effective_crr(), WHEEL_BEARING["roll_cut_frac"] * 100),
        "EVERY speed. Rolling load dominates at low speed, so this is a steady, always-",
        "on MPG gain (biggest in slow/urban driving) that stacks with the low-Crr tyres.",
        "The flywheel + core run on the same passive-magnetic principle (<0.3%% drag).",
    ]),
    ("8d. THROUGH-BODY FLOW DUCT (kill the rear vacuum / wake drag)", [
        "A wide, thin, STRAIGHT hollow duct runs from a low front-GRILL intake, 100%%",
        "straight through the body, to a rear DIFFUSER. It lets high-pressure nose air",
        "flow right through the car and exit into the low-pressure WAKE behind it,",
        "'filling' that rear vacuum pocket. That attacks PRESSURE (form) drag -- the",
        "DOMINANT drag source above ~50 mph -- instead of just skin friction.",
        "The rear diffuser widens like a VENTURI to slow the exit air smoothly and",
        "recover pressure, so the wake is calmer and the car sheds a big turbulent",
        "low-pressure region. Scaled straight-through pipe (%.0f x %.0f mm, ~%.1f m)" % (
            DIMS["flow_duct_width_mm"], DIMS["flow_duct_height_mm"],
            DIMS["flow_duct_len_mm"] / 1000.0),
        "with front slots + a rear diffuser (a real part in the assembly -- hover it).",
        "Effect: it cuts the effective Cd ~%.0f%% at low speed rising to ~%.0f%% at" % (
            DUCT["base"] * DUCT["blend"] * 100, (DUCT["base"] + DUCT["speed"]) * DUCT["blend"] * 100),
        "highway speed (the wake vacuum grows with speed), folded straight into the",
        "Cd so the M chart AND drive economy both show it -- an estimated 12-20%% MPG",
        "gain at highway speed. It pairs with the boat-tail, morphing aero, hydrophobic",
        "skin and drag-capture grooves: the skin slips most air, the grooves catch the",
        "remainder, and the duct MOVES the useful air through -- so the car behaves",
        "like a near-perfect streamlined body. Parked, the duct doubles as passive",
        "cabin / battery ventilation. (In crosswinds the intake can partly close for",
        "stability, trading a little of the gain for control.)",
    ]),
    ("8e. WHOLE-CAR VIEW + REAL-LIFE PHYSICS (press TAB to reach it)", [
        "TAB now cycles THREE views: ENGINE PREVIEW > CAR VIEW > DRIVE. The CAR VIEW is",
        "a SEPARATE 3D model of the ENTIRE vehicle, built to real-life scale from the",
        "CAR dimensions -- %.0f x %.0f x %.0f mm, wheelbase %.0f, track %.0f mm." % (
            CAR["length_mm"], CAR["width_mm"], CAR["height_mm"],
            CAR["wheelbase_mm"], CAR["track_mm"]),
        "It has its OWN full / exploded / assembly views + section-cut, and every",
        "component is present and hoverable: the carbon monocoque body + hydrophobic",
        "skin, glass cabin, structural battery floor, supercap bank, four separate",
        "EM rail-ring wheel drives on passive magnetic bearings, the rotary range-",
        "extender engine + tungsten flywheel, the steam boiler, the solar roof, the windshield sun-lens,",
        "the through-body flow duct, the parked windmill fin, the regen suspension",
        "dampers and the passenger pedal generators. Wheels/engine/flywheel/windmill",
        "spin; EXPLODED (key 2) fans the whole car apart; SECTION-CUT (4/X) slices the",
        "body so the battery + engine read inside. Orbit/zoom/pan and hover just like",
        "the engine view.",
        "Each wheel drive shows the rim-rotor/flywheel disc inside the wheel, bonded",
        "magnet band, stationary rail stator, exaggerated visible air-gap ring, copper",
        "winding, 96-pole markers and passive PM bearing hub; the front-right wheel",
        "adds detail callouts as the legend while all four carry the same geometry.",
        "REAL-LIFE PHYSICS: the drive + M-chart use textbook ROAD-LOAD -- aero force",
        "0.5*rho*Cd*A*v^2 (power ~ v^3) + rolling Crr*m*g (power ~ v^1). The model is",
        "cross-checked to the car: frontal area A = width x height x fill = %.2f x %.2f" % (
            CAR["width_mm"] / 1000.0, CAR["height_mm"] / 1000.0),
        "x %.2f = %.2f m^2 (~VEH %.2f), mass %.0f kg, Cd %.2f (morphed/duct/skin lower" % (
            CAR["frontal_fill"],
            CAR["width_mm"] / 1000.0 * CAR["height_mm"] / 1000.0 * CAR["frontal_fill"],
            VEH["frontal_area_m2"], VEH["curb_mass_kg"], VEH["Cd"]),
        "it further), Crr %.4f (magnetic bearings lower it). So the MPG numbers come" % VEH["Crr"],
        "from the SAME physical dimensions you see in the CAR VIEW -- not arbitrary.",
    ]),
    ("9. MANUAL ENGINE TEST BENCH (preview, drag sliders)", [
        "Every input is manual so the engine can be tested. Drag the left-panel",
        "sliders: ENGINE POWER, HYDRAULIC PRESSURE, MAIN WET CLUTCH (grip), TRANS",
        "WET CLUTCH (bind), COOLANT FLOW and SET GEAR (ring shift). Both clutches",
        "only bite with hydraulic pressure.",
        "Live kinematics: piston -> main clutch (slip) -> core -> transmission gear",
        "wet-clutch bind (blends 1:1 up to the gear ratio) -> output gear -> generator.",
        "Every part/tooth/screw turns at its TRUE RPM (slipping plates + ring layers",
        "spin at different rates). Readouts show RPM, ratio, slip %, boost pressure,",
        "air/fuel flow, hydraulic clamp force, friction heat, cooling, BLOCK TEMP,",
        "thermal state, heat removed and main+ORC electric.",
    ]),
    ("10. WATCHING IT WORK (playback)", [
        "The preview always shows fuel injection, combustion and rotation to scale.",
        "P pauses; - / = step the view speed (down to 0.05x) so you can watch a",
        "single chamber inject, then combust at the fixed injector frame by frame.",
        ", and . step combustions/rev anywhere from 1 to 8 (or drag the",
        "COMBUSTIONS/REV bar); [ / ] add/remove pistons (1-10).",
    ]),
    ("11. EM RAIL-RING DIRECT WHEEL DRIVE (no separate wheel motor)", [
        "A NEW direct electromagnetic rail drive turns the wheels. There is NO separate",
        "wheel motor or electric engine: a stationary electromagnetic RAIL RING surrounds",
        "each wheel like a stator field around a rotor -- and the WHEEL RIM ITSELF is the",
        "ROTOR, bonded as ONE hard-locked piece (no clutch in the wheel), floating on a",
        "MAGNETIC BEARING and extended just under an inch to clear the rail.",
        "Energising the %d-pole hybrid superconducting + permanent-magnet rail (0-%.0f A," % (
            RAIL_DRIVE["poles"], RAIL_DRIVE["coil_amp_max"]),
        "ECU-modulated) magnetically spins/locks the rim with scalable, GEARLESS torque,",
        "and the same rail REGENERATES on braking. The full standalone unit is a %.1f m," % RAIL_DRIVE["disc_dia_m"],
        "~%.0f kg disc whose inner rim is a FLYWHEEL kinetic store to ~%.0f rpm, clutched" % (
            RAIL_DRIVE["unit_kg"], RAIL_DRIVE["flywheel_rpm"]),
        "to the output by a single-plate wet clutch (~%.0fk Nm cont / ~%.0fk Nm burst);" % (
            RAIL_DRIVE["cont_torque_nm"] / 1000, RAIL_DRIVE["burst_torque_nm"] / 1000),
        "in the car it is the LIGHT rim-rotor derivative on each wheel.",
        "MODEL DETAIL: CAR VIEW now breaks the drive into four labelled, hoverable",
        "wheel modules -- front/rear left/right -- and each module contains the visible",
        "in-rim rotor/flywheel disc, bonded rotor magnet band, stationary EM rail,",
        "0.15 mm rail air gap, copper rail winding, pole blocks and PM bearing hub.",
        "WHY IT LIFTS MPG: no gearbox and no drive clutch in the path means very low",
        "loss -- the drivetrain efficiency rises to ~%.1f%% (from %.0f%%) and regen to" % (
            VEH["drivetrain_eff"] * 100, 95),
        "~%.0f%%, so LESS battery energy is spent per mile and MORE is recovered. The" % (VEH["regen_frac"] * 100),
        "wheels + a standalone cross-section unit are both in the CAR VIEW (hover them).",
    ]),
    ("12. SYSTEM OPTIMIZATION + SCALING (why the arrangement is synergistic)", [
        "The whole energy system was SIZE-optimized by a search harness",
        "(optimize_harvest.py) that ran 300,000+ configurations. It is an honest",
        "engineering optimizer, NOT a free-energy cranker: sizing any harvester UP also",
        "adds MASS (hurts rolling + accel) and, for external parts, DRAG (hurts aero as",
        "v^3), so every knob has a real INTERIOR optimum -- 'sized to scale'. All",
        "efficiencies are bounded by real tech ceilings; energy conservation is enforced",
        "(the steam loop makes ZERO watts at ambient -- no perpetual motion).",
        "WHAT IT FOUND (net cycle-MPG optimum):",
        " - MAX the low-mass / high-value paths: solar (~%.0f W), drag-air capture," % (
            (AMBIENT_HARVEST_W["solar"] + AMBIENT_HARVEST_W["solar_roof"]) * SUN),
        "   and the loss-free EFFICIENCIES -- rail drivetrain to ~%.0f%%, regen to ~%.0f%%," % (
            VEH["drivetrain_eff"] * 100, VEH["regen_frac"] * 100),
        "   heat-recovery to its Carnot-bounded %.0f%% cap." % (THERM["recovery_eff_cap"] * 100),
        " - TRIM the heavy paths to their sweet spot: the pendulum bobs, the suspension",
        "   peak and the thermoelectric film all cost mass, so oversizing them LOSES MPG",
        "   -- the optimizer sized each to where its benefit still beats its mass.",
        "WHY IT IS SYNERGISTIC: each harvester owns a DIFFERENT regime, so together they",
        "blanket every condition with no gap and no overlap waste --",
        "   SOLAR: always-on (even parked, in sun).  DRAG-CAPTURE: grows with speed (v^3).",
        "   SUSPENSION/PENDULUM: from MOTION + stop-go + corners + inclines.",
        "   STEAM/TEG/PCM: from engine + sun HEAT (and after shutdown).",
        "   REGEN + FLYWHEEL: from braking + downhill.   WINDMILL: from wind when parked.",
        "The result: +155.4%% drive-cycle MPG for ~51.7 kg of added optimization mass.",
        "0-LOSS is the ASYMPTOTIC target (thermodynamics forbids reaching it); the design",
        "is pushed to the realistic ceiling of each path, then STOPPED where physics says",
        "stop. Re-run optimize_harvest.py any time to re-solve the sizing.",
    ]),
    ("13. KINETIC->ELECTRICAL EFFICIENCY CHAIN (fuel -> wheel, the 0-loss gap)", [
        "The generator path -- turning combustion KINETIC energy into ELECTRICITY -- was",
        "re-optimized to the loss-free ceiling of each step. Following one unit of fuel:",
        "  combustion -> indicated work   %4.1f%%   (the %.0f%% lost here is HEAT --" % (
            PHYS["indicated_eff"] * 100, (1 - PHYS["indicated_eff"]) * 100),
        "                                          thermodynamically unavoidable, but the",
        "                                          steam loop claws much of it back)",
        "  -> mechanical (shaft)          %4.1f%%   (magnetic bearings, %.1f%% step loss)" % (
            PHYS["indicated_eff"] * PHYS["mechanical_eff"] * 100,
            (1 - PHYS["mechanical_eff"]) * 100),
        "  -> GENERATOR (kinetic->elec)   %4.1f%%   (axial-flux PM, only %.1f%% step loss)" % (
            PHYS["indicated_eff"] * PHYS["mechanical_eff"] * PHYS["generator_eff"] * 100,
            (1 - PHYS["generator_eff"]) * 100),
        "  -> gearless rail coupling      %4.1f%%   (%.1f%% step loss)" % (
            PHYS["indicated_eff"] * PHYS["mechanical_eff"] * PHYS["generator_eff"]
            * PHYS["gear_mesh_eff"] * 100, (1 - PHYS["gear_mesh_eff"]) * 100),
        "  + STEAM heat-recovery add-back        => FUEL -> ELECTRICITY ~48%",
        "  x EM rail drivetrain %.0f%%              => FUEL -> WHEEL ~%.1f%%" % (
            VEH["drivetrain_eff"] * 100, FUEL_TO_WHEEL_EFF * 100),
        "The generator step alone was lifted to %.1f%% (from 95%%), raising its output" % (
            PHYS["generator_eff"] * 100),
        "~%.0f%% per gallon -- so the range-extender makes MORE charge per burst, fires" % 6.8,
        "FEWER/shorter bursts, and burns less fuel. Every step above the combustion node",
        "is near its physical ceiling; the big remaining 'loss' is the combustion heat",
        "itself, which is why the CLOSED-LOOP STEAM recovery (2b) matters so much -- it",
        "is the only way to push fuel->wheel toward the 0-loss target past the ~44%",
        "thermal wall. Bounded: recovered electricity is ALWAYS < the heat drawn.",
    ]),
    ("14. STORAGE + ROUTING SYNERGY (real round-trips, best arrangement)", [
        "CORRECTNESS: real storage is NOT lossless -- every joule STORED then retrieved",
        "pays a round-trip. So each flow is ROUTED through the most efficient buffer that",
        "can take it (the loss is taken on the way IN, so charge-then-discharge NEVER",
        "returns more than went in -- energy-honest, no free storage):",
        " - SUPERCAPS (round-trip %.0f%%): take the FAST charge/drain spikes -- braking," % (
            STORAGE["supercap_rt"] * 100),
        "   downhill and engine bursts -- so the battery never sees a hard pulse.",
        " - BATTERY (round-trip %.0f%%): bulk / slow storage of the steady surplus." % (
            STORAGE["battery_rt"] * 100),
        " - FLYWHEEL (round-trip %.0f%%): mechanical OVERFLOW when both are full/empty." % (
            STORAGE["flywheel_rt"] * 100),
        "WHY THE ARRANGEMENT IS OPTIMAL: routing the fast cycling through the near-",
        "lossless supercaps keeps the effective round-trip ~96%%, far above cycling it",
        "all through the battery. The optimizer (300k+ tests) also found the supercap",
        "should be sized for PEAK POWER, NOT bulk energy -- a bigger bank barely improves",
        "the round-trip but supercaps are HEAVY (~125 kg/kWh), so oversizing LOSES MPG.",
        "It is therefore held to %.2f kWh (peak-buffering), while the %.0f kWh structural" % (
            ELEC["supercap_kwh"], ELEC["batt_kwh"]),
        "battery does the bulk. Headroom is still king: the %.0f-%.0f%% SOC window leaves" % (
            ELEC["soc_min"] * 100, ELEC["soc_max"] * 100),
        "room to swallow every regen + harvest pulse WITHOUT overflowing to the lossy",
        "flywheel path. Net: modelling the (previously ignored) storage loss is a small",
        "MPG cost the SMART ROUTING almost entirely recovers -- correctness + synergy.",
        "ONE-PASS NETTING: harvest, generator, steam and regen are netted against the",
        "traction demand in a SINGLE routing pass, so any watt used the SAME instant for",
        "traction flows STRAIGHT to the wheel rails and never pays a storage round-trip -- only",
        "the true surplus is buffered. While cruising that alone keeps ~%.0f W of harvest" % 40,
        "loss-free every second (it was being stored then immediately re-drawn).",
    ]),
    ("15. OPERATING-POINT + CONTROL ANALYSIS (why it is already optimal)", [
        "The last lever is COORDINATION, checked by optimize_control.py:",
        " - OPERATING POINT: 3,288 firing points (RPM x combustions/rev) were swept",
        "   through the REAL engine physics, then heat recovery was bounded to actual",
        "   same-second heat flow so stored boiler heat is not double-counted.",
        "   Result: %.0f RPM x 8/rev gives %.1f%% net fuel->electricity vs %.1f%% before." % (
            FIRING_GEN_RPM, 52.9, 52.2),
        " - CONTROL POLICY: swept the SOC window + burst thresholds over a DEMANDING",
        "   run (85 mph, hills, low sun 0.05, low initial SOC). Result: the current",
        "   fire threshold was already right; lowering the stop target to %.0f%% keeps" % (
            ENGINE_OFF_SOC * 100),
        "   more regen headroom. Stress case: ~738 MPG, engine-on ~17.8%%; favorable",
        "   routes are much higher because road load is tiny and harvest covers more.",
        "CONCLUSION: across generation, conversion, storage AND control, every path now",
        "sits at its physics-bounded ceiling; the only remaining 'losses' are the ones",
        "thermodynamics forbids removing (combustion heat -- partly reclaimed by steam)",
        "and the tiny, correct storage round-trip the one-pass routing minimises. The",
        "system is at its engineered OPTIMUM. Re-run the two optimizer scripts anytime.",
    ]),
    ("CONTROLS", [
        "TAB ... cycle ENGINE PREVIEW > CAR VIEW > DRIVE   M ... MPG   I ... info   H ... help",
        "CAR VIEW: the WHOLE vehicle to real-life scale -- its own 1/2/3 + 4/X views (8e)",
        "PREVIEW/CAR VIEWS:  1 = FULL   2 = EXPLODED   3 = ASSEMBLY",
        "SECTION-CUT:  4 or X = toggle the cross-section over FULL / EXPLODED",
        "PREVIEW/CAR: drag orbit | wheel zoom-at-cursor | right/middle pan | L labels chip",
        "         hover a part = read spec card | click = pin / unpin selection",
        "TEST BENCH: drag the left sliders (power, hydraulic, main clutch, bind, gear)",
        "PLAYBACK: P pause | - / = view speed | [ ] pistons 1-10 | , . combustions 1-8",
        "ASSEMBLY: click or N = place next part | B = back | F = all | 0 = clear",
        "DRIVE:  W/Up throttle | S/Down brake | A/D steer | C cruise | E engine burst",
        "        R = Drive <-> Reverse (stopped) | G downhill | SPACE brake",
        "        K = passenger PEDAL-ASSIST | F = boiler working fluid",
        "ESC ........ quit",
    ]),
]


# =============================================================================
# SECTION 7 -- HUD / UI HELPERS
# =============================================================================

def vgradient(surf, top, bot):
    h = surf.get_height()
    w = surf.get_width()
    for y in range(h):
        t = y / max(1, h)
        col = (int(top[0] + (bot[0] - top[0]) * t),
               int(top[1] + (bot[1] - top[1]) * t),
               int(top[2] + (bot[2] - top[2]) * t))
        pygame.draw.line(surf, col, (0, y), (w, y))


def bar(surf, font, x, y, w, h, frac, color, label, valtext, lo=None, hi=None):
    pygame.draw.rect(surf, C_PANEL_HI, (x, y, w, h), border_radius=4)
    frac = max(0.0, min(1.0, frac))
    pygame.draw.rect(surf, color, (x, y, int(w * frac), h), border_radius=4)
    if lo is not None:
        pygame.draw.line(surf, C_WARN, (x + int(w * lo), y - 2), (x + int(w * lo), y + h + 2), 1)
    if hi is not None:
        pygame.draw.line(surf, C_WARN, (x + int(w * hi), y - 2), (x + int(w * hi), y + h + 2), 1)
    surf.blit(font.render(label, True, C_TEXT_DIM), (x, y - 16))
    img = font.render(valtext, True, C_TEXT)
    surf.blit(img, (x + w - img.get_width(), y - 16))


def panel(surf, x, y, w, h, alpha=210):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((C_PANEL[0], C_PANEL[1], C_PANEL[2], alpha))
    surf.blit(s, (x, y))
    pygame.draw.rect(surf, C_PANEL_HI, (x, y, w, h), 1, border_radius=6)


def wrap_text(font, text, maxpx):
    """Greedy word-wrap a string to a pixel width; returns a list of lines."""
    out, cur = [], ""
    for word in text.split(" "):
        trial = word if not cur else cur + " " + word
        if font.size(trial)[0] <= maxpx:
            cur = trial
        else:
            if cur:
                out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out or [""]


# =============================================================================
# SECTION 8 -- APPLICATION
# =============================================================================

class App:
    MODE_PREVIEW = 0        # engine 3D view (test bench)
    MODE_DRIVE = 1          # drive the car
    MODE_CAR = 2            # whole-vehicle 3D view (full / exploded / assembly)

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("GmansRun V1.17 -- HOHEV-Rotary Gen 4 Digital Twin")
        self.W, self.H = 1600, 920
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        mono = "consolas,dejavusansmono,monospace"
        self.font = pygame.font.SysFont(mono, 16)
        self.fbig = pygame.font.SysFont(mono, 30, bold=True)
        self.fhuge = pygame.font.SysFont(mono, 64, bold=True)
        self.fsmall = pygame.font.SysFont(mono, 13)

        self.renderer = EngineRenderer()
        # separate WHOLE-CAR renderer (its own full/exploded/assembly + section views),
        # framed for the ~3.8 m vehicle instead of the engine.
        self.car = EngineRenderer(parts_builder=build_car_parts, supports_pistons=False,
                                  home_az=0.85, home_el=-0.34, home_dist=6.2)
        # the SAME whole-car model, rendered from a fixed 3rd-person chase camera as
        # the drivable vehicle in DRIVE mode (its wheels spin on the shared 'wheel'
        # group angle set by the live road speed). Built at REDUCED mesh detail --
        # the extra facets are invisible at chase distance but ~halve the per-frame
        # cost, protecting the interactive drive-loop frame rate.
        global VISUAL_DETAIL
        _vd = VISUAL_DETAIL
        VISUAL_DETAIL = 0.7
        self.drivecar = EngineRenderer(parts_builder=build_car_parts, supports_pistons=False,
                                       home_az=math.pi + 0.22, home_el=-0.16, home_dist=5.4)
        VISUAL_DETAIL = _vd
        self.pt = Powertrain()
        self.world = DriveWorld()

        self.mode = self.MODE_PREVIEW
        self.engine_angle = 0.0
        self.paused = False
        self.speeds = [0.05, 0.15, 0.5, 1.0]
        self.speed_idx = 3
        self.combustions = DRIVE_COMBUSTIONS       # 1..8 combustions/rev; 8 is drive optimum
        self._eff_comb = DRIVE_COMBUSTIONS         # combustions actually fired now
        self._drag_combust = False                 # dragging the combustion bar
        self.fire_mode = "SEQUENTIAL"              # fixed mechanical timing
        self._injection = np.zeros(DIMS["chambers"])
        # --- manual engine test-bench inputs (all mouse-adjustable) --------
        self.engine_power = 0.6                    # 0..1 piston drive
        self.hyd_pressure = 1.0                    # hydraulic supply pressure
        self.clutch_main = 1.0                     # 1st (main) wet clutch command
        self.trans_bind_cmd = 1.0                  # transmission ring-bind command
        self.coolant_flow = 0.7                    # coolant pump flow
        self.gear_sel = 0                          # selected transmission ring
        self.piston_rpm = 0.0
        self.fly_rpm_v = 0.0
        self.supercharger_engage = 0.0
        self.group_angle = {"default": 0.0}        # live angle per kinematic group
        self.kin = {"piston": 0, "core": 0, "gen": 0, "out": 0, "trans_out": 0,
                    "ratio": 1.0, "gear": 1, "ngear": DIMS["trans_rings"], "gr": 1.0,
                    "slip1": 0, "bind_slip": 0, "fly": 0, "ep": 0, "c1": 0, "bind": 0,
                    "combust_min": 0, "super": 0.0, "boost": 1.0}
        # thermal + heat-recovery state
        self.temp_c = THERM["ambient_c"]
        self.boiler_c = THERM["ambient_c"]         # boiler working-fluid temperature
        self.pcm_c = THERM["ambient_c"]            # phase-change heat-bank temperature
        self.fluid = DEFAULT_FLUID                  # selected closed-loop fluid
        self.fluid_kwh = 0.0                         # recovered energy on this charge
        self.fluid_life = 1.0                        # 1..0 until a fluid change is due
        self.fluid_miles0 = 0.0                      # odometer at last fluid change
        self.pedal_on = False                        # passenger pedal-assist engaged
        self.pedal_engage = 0.0                      # ramp 0..1
        self.pedal_wh = 0.0                          # cumulative pedal energy (Wh)
        self.ambient_kw = 0.0                        # live multi-layer ambient harvest
        self.harvest_wh = 0.0                        # cumulative ambient harvest (Wh)
        self.teg_kw = 0.0                            # live thermoelectric harvest
        self.susp_kw = 0.0                           # live dynamic suspension harvest
        self.steer_kw = 0.0                          # live regenerative-steering harvest
        self.solar_boiler_kw = 0.0                   # live windshield-lens heat to boiler
        self.solar_roof_kw = 0.0                     # live dedicated PV-roof electric
        self.drag_kw = 0.0                           # live passive drag-air capture electric
        self.wind_kw = 0.0                           # live parked-windmill electric
        self.pend_kw = 0.0                           # live inertial-pendulum-rim electric
        self.pend_deploy = 1.0                       # skirt deploy 0..1 (retracts at speed)
        self._prev_v = 0.0                           # last frame speed (m/s) for accel
        self._prev_grade = 0.0                       # last frame grade for incline-change rate
        self.extra_wh = 0.0                          # cumulative bonus-harvest Wh (all paths)
        self.cd_eff = VEH["Cd"]                      # live morphed drag coefficient
        self.soc_target = ENGINE_OFF_SOC             # live predictive SOC charge target
        self.future_grade = 0.0                      # predictive look-ahead grade
        self.therm = {"temp": self.temp_c, "heat_kw": 0.0, "cool_kw": 0.0,
                      "orc_kw": 0.0, "main_kw": 0.0, "total_kw": 0.0,
                      "steam_kw": 0.0, "steam_bar": 0.0, "boiler_c": self.boiler_c,
                      "exh2_kw": 0.0}
        self.phys = {"pressure_kpa": PHYS["ambient_pa"] / 1000.0,
                     "pressure_ratio": 1.0, "chamber_cc": effective_chamber_volume_m3() * 1e6,
                     "events_s": 0.0, "air_g_s": 0.0, "fuel_g_s": 0.0,
                     "afr": PHYS["target_afr"], "gross_kw": 0.0, "shaft_kw": 0.0,
                     "gen_kw": 0.0, "waste_kw": 0.0, "block_heat_kw": 0.0,
                     "exhaust_kw": 0.0, "friction_kw": 0.0, "cool_kw": 0.0,
                     "orc_kw": 0.0, "hyd_bar": 0.0, "main_clamp_kn": 0.0,
                     "ring_clamp_kn": 0.0, "bearing_kw": 0.0, "derate": 1.0,
                     "overheat": False, "shutdown": False, "status": "NORMAL",
                     "gear_eff": 1.0, "heat_in_block": 0.0, "exh_turbo_kw": 0.0,
                     "exh_stage2_kw": 0.0, "exh_to_boiler_kw": 0.0, "steam_kw": 0.0,
                     "steam_bar": 0.0, "steam_max_bar": FLUIDS[DEFAULT_FLUID]["max_bar"],
                     "steam_ramp": 0.0, "steam_draw_kw": 0.0, "recovery_eff": 0.0,
                     "fluid": DEFAULT_FLUID, "boiler_c": THERM["ambient_c"],
                     "heat_in_boiler": 0.0, "shield_flow_kw": 0.0,
                     "passive_kw": 0.0, "condition_kw": 0.0}
        self._drag_slider = None
        self.show_info = False
        self.show_mpg = False                        # real-life MPG-vs-speed chart
        self.pedal_extend = 0.0                      # pop-out pedals extended 0..1
        self.show_help = True
        self.show_labels = False                     # per-part name callouts OFF by default
        #                                              (clean view); L / chip toggles them on
        self._labels_chip = None                     # clickable LABELS toggle rect
        self.info_scroll = 0
        self.dragging = False
        self.panning = False
        self._press_pos = None
        self.throttle = 0.0
        self.brake = 0.0
        self.force_grade = 0.0
        self.cur_grade = 0.0
        self.cur_mph = 0.0
        self.running = True
        self._frame = 0                              # global frame counter
        self._inset_cache = None                     # cached LIVE-ENGINE drive inset
        self.elec = solve_electrical(0.0)            # live HV-bus electrical state
        self.bg = None
        self._rebuild_bg()

    def _rebuild_bg(self):
        self.bg = pygame.Surface((self.W, self.H))
        vgradient(self.bg, BG_TOP, BG_BOT)
        self._sky = pygame.Surface((self.W, self.H))   # cached drive sky gradient
        vgradient(self._sky, C_SKY1, C_SKY2)

    def _preview_like(self):
        """True in either 3D-inspection mode (engine PREVIEW or whole-CAR view)."""
        return self.mode in (self.MODE_PREVIEW, self.MODE_CAR)

    def _rend(self):
        """The renderer the shared camera/view controls act on: the whole-car
        renderer in CAR mode, otherwise the engine renderer."""
        return self.car if self.mode == self.MODE_CAR else self.renderer

    # ---------------------------------------------------------------- events
    def handle_events(self, dt):
        keys = pygame.key.get_pressed()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE:
                self.W, self.H = max(1120, e.w), max(700, e.h)
                self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
                self._rebuild_bg()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_TAB:
                    self.mode = {self.MODE_PREVIEW: self.MODE_CAR,
                                 self.MODE_CAR: self.MODE_DRIVE,
                                 self.MODE_DRIVE: self.MODE_PREVIEW}[self.mode]
                elif e.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif e.key == pygame.K_m:
                    self.show_mpg = not self.show_mpg      # real-life MPG chart
                elif e.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif e.key == pygame.K_l:
                    self.show_labels = not self.show_labels
                elif e.key == pygame.K_r and self._preview_like():
                    self._rend().reset_view()
                elif self._preview_like() and e.key in (
                        pygame.K_1, pygame.K_2, pygame.K_3):
                    self._rend().set_view({pygame.K_1: "full", pygame.K_2: "exploded",
                                           pygame.K_3: "assembly"}[e.key])
                elif self._preview_like() and e.key in (pygame.K_4, pygame.K_x):
                    self._rend().toggle_section()      # section CUT is a toggle
                elif self._preview_like() and self._rend().view == "assembly" \
                        and e.key in (pygame.K_n, pygame.K_SPACE, pygame.K_RIGHT):
                    self._rend().assembly_next()
                elif self._preview_like() and self._rend().view == "assembly" \
                        and e.key in (pygame.K_b, pygame.K_LEFT, pygame.K_BACKSPACE):
                    self._rend().assembly_prev()
                elif self._preview_like() and self._rend().view == "assembly" \
                        and e.key == pygame.K_f:
                    self._rend().assembly_all()
                elif self._preview_like() and self._rend().view == "assembly" \
                        and e.key == pygame.K_0:
                    self._rend().assembly_clear()
                elif e.key == pygame.K_p and self._preview_like():
                    self.paused = not self.paused
                elif e.key == pygame.K_MINUS and self._preview_like():
                    self.speed_idx = max(0, self.speed_idx - 1)
                    self.paused = False
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS) and self._preview_like():
                    self.speed_idx = min(len(self.speeds) - 1, self.speed_idx + 1)
                    self.paused = False
                elif e.key == pygame.K_LEFTBRACKET and self.mode == self.MODE_PREVIEW:
                    self.renderer.set_pistons(self.renderer.n_pistons - 1)
                elif e.key == pygame.K_RIGHTBRACKET and self.mode == self.MODE_PREVIEW:
                    self.renderer.set_pistons(self.renderer.n_pistons + 1)
                elif e.key == pygame.K_COMMA and self.mode == self.MODE_PREVIEW:
                    self.combustions = max(1, self.combustions - 1)
                elif e.key == pygame.K_PERIOD and self.mode == self.MODE_PREVIEW:
                    self.combustions = min(DIMS["chambers"], self.combustions + 1)
                elif e.key == pygame.K_f:
                    i = (FLUID_ORDER.index(self.fluid) + 1) % len(FLUID_ORDER)
                    self.fluid = FLUID_ORDER[i]      # cycle boiler working fluid
                    self.fluid_kwh = 0.0             # fresh charge -> reset service
                    self.fluid_miles0 = self.pt.miles
                elif e.key == pygame.K_k:
                    self.pedal_on = not self.pedal_on   # passenger pedal-assist
                elif e.key == pygame.K_r and self.mode == self.MODE_DRIVE:
                    if abs(self.world.v) < 0.6:        # shift D <-> R when stopped
                        self.world.gear *= -1
                        self.world.cruise = False
                elif e.key == pygame.K_c and self.mode == self.MODE_DRIVE:
                    self.world.cruise = not self.world.cruise
                    self.world.cruise_set = self.world.v
                elif e.key == pygame.K_e:
                    self.pt.force_timer = 6.0      # forced 6s engine burst
                elif e.key == pygame.K_g and self.mode == self.MODE_DRIVE:
                    self.force_grade = -0.06 if self.force_grade == 0 else 0.0
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if self.show_info:
                    continue
                if e.button == 1:
                    # click the LABELS toggle chip (engine preview + car view)
                    if self._preview_like() and self._labels_chip \
                            and self._labels_chip.collidepoint(e.pos):
                        self.show_labels = not self.show_labels
                        continue
                    if self.mode == self.MODE_PREVIEW and self._combust_hit(e.pos):
                        self._drag_combust = True
                        self._combust_set_from_x(e.pos[0])
                        continue
                    hit = self._bench_hit(e.pos) if self.mode == self.MODE_PREVIEW else None
                    if hit is not None:
                        self._drag_slider = hit
                        self._bench_set_from_x(hit, e.pos[0])
                    else:
                        self.dragging = True
                        self._press_pos = e.pos
                elif e.button in (2, 3):
                    self.panning = True
                elif e.button in (4, 5) and self._preview_like():
                    rect = pygame.Rect(0, 34, self.W, self.H - 34)
                    factor = 0.82 if e.button == 4 else 1.22
                    self._rend().zoom_at(factor, e.pos, rect)
            elif e.type == pygame.MOUSEBUTTONUP:
                if (e.button == 1 and self._drag_slider is None and self._press_pos is not None
                        and self._preview_like() and not self.show_info):
                    moved = math.hypot(e.pos[0] - self._press_pos[0],
                                       e.pos[1] - self._press_pos[1])
                    if moved < 6:
                        self._preview_click()
                self._press_pos = None
                self._drag_slider = None
                self._drag_combust = False
                self.dragging = self.panning = False
            elif e.type == pygame.MOUSEMOTION:
                dx, dy = e.rel
                if self._drag_combust:
                    self._combust_set_from_x(e.pos[0])
                elif self._drag_slider is not None:
                    self._bench_set_from_x(self._drag_slider, e.pos[0])
                elif self._preview_like() and not self.show_info:
                    fine = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    if self.dragging:
                        self._rend().orbit(dx, dy, fine=fine)
                    elif self.panning:
                        self._rend().pan_by(dx, dy, fine=fine)
            elif e.type == pygame.MOUSEWHEEL:
                if self.show_info:
                    self.info_scroll = max(0, self.info_scroll - e.y * 30)
                elif self._preview_like():
                    rect = pygame.Rect(0, 34, self.W, self.H - 34)
                    factor = 0.82 ** e.y if e.y > 0 else 1.22 ** (-e.y)
                    self._rend().zoom_at(factor, pygame.mouse.get_pos(), rect)

        if self.mode == self.MODE_DRIVE and not self.show_info:
            t_press = keys[pygame.K_w] or keys[pygame.K_UP]
            b_press = keys[pygame.K_s] or keys[pygame.K_DOWN] or keys[pygame.K_SPACE]
            self.throttle += ((1.0 if t_press else 0.0) - self.throttle) * min(1, dt * 6)
            self.brake += ((1.0 if b_press else 0.0) - self.brake) * min(1, dt * 8)
            self.world.steer = ((1 if keys[pygame.K_d] else 0)
                                - (1 if keys[pygame.K_a] else 0))

    def _preview_click(self):
        """Click in preview / car view: pin/unpin a part (full/exploded) or place
        the next part (assembly puzzle) on the ACTIVE renderer."""
        r = self._rend()
        if r.view == "assembly":
            r.assembly_next()
        else:
            h = r.hovered
            r.selected = None if (h is None or r.selected == h) else h

    def _compute_firing(self, active, combustions, mode):
        """Set per-chamber injection + combustion intensity from the current
        rotation angle (deterministic, so it freezes correctly when paused).
        Mechanical timing is fixed: either one chamber fires per revolution or
        all eight chambers fire once each per revolution."""
        glow = self.pt.firing_glow
        inj = self._injection
        glow[:] = 0.0
        inj[:] = 0.0
        if not active:
            return
        nch = DIMS["chambers"]
        combustions = combustion_count(combustions)
        scheduled = []
        for j in range(combustions):
            idx = int(round(j * nch / combustions)) % nch
            if idx not in scheduled:
                scheduled.append(idx)
        while len(scheduled) < combustions:
            for idx in range(nch):
                if idx not in scheduled:
                    scheduled.append(idx)
                    break

        def window(d):
            # wider trailing glow so several chambers read as "firing" at once
            comb = max(0.0, 1.0 - d / 1.5) if -0.06 < d < 1.5 else 0.0
            ij = max(0.0, 1.0 - (abs(d) - 0.06) / 0.54) if -0.6 < d <= -0.06 else 0.0
            return comb, ij

        for i in scheduled:
            base = i * 2 * math.pi / nch
            d = ((self.engine_angle + base - INJECTOR_ANGLE + math.pi)
                 % (2 * math.pi)) - math.pi    # 0 exactly at the injector
            glow[i], inj[i] = window(d)

    def _kinematics(self, dt, drive):
        """Live drivetrain kinematics: piston -> main clutch (slip) -> core ->
        transmission ring-bind selector -> output gear -> generator.
        Advances every group's angle by its true RPM, and records readouts."""
        nrings = DIMS["trans_rings"]
        ratios = trans_ratios()
        if drive:
            moving = abs(self.world.v) > 0.3
            ep = 1.0 if self.pt.engine_on else (0.35 if moving else 0.12)
            c1 = 1.0 if self.pt.engine_on else (0.4 if moving else 0.0)
            bind = self.pt.trans_bind
            sc = self.pt.supercharger_engage
            gear = min(nrings - 1, int(abs(self.world.v) / 6.0))
        else:
            ep = self.engine_power
            # clutches engage ONLY via hydraulic pressure (command x supply)
            c1 = self.clutch_main * self.hyd_pressure
            bind = self.trans_bind_cmd * self.hyd_pressure
            target_sc = 1.0 if ep > 0.02 else 0.0
            self.supercharger_engage += (target_sc - self.supercharger_engage) \
                * min(1.0, dt * SUPERCHARGER_RAMP_RATE)
            sc = self.supercharger_engage
            gear = max(0, min(nrings - 1, int(self.gear_sel)))

        duty = combustion_duty(self._eff_comb)
        boost = boost_multiplier(sc)
        target = ep * duty * MAX_PISTON_RPM * (0.96 + 0.04 * boost)
        self.piston_rpm += (target - self.piston_rpm) * min(1.0, dt * 1.5)
        core = self.piston_rpm * c1                 # main clutch slip (piston->core)
        gr = ratios[gear]
        eff = 1.0 + (gr - 1.0) * bind               # ring-bind blends 1:1 -> gear
        trans_out = core * eff                      # bound transmission output
        selected_inner = trans_ring_edges()[gear][0] + 0.006
        rog = DIMS["out_gear_d_mm"] * 0.001 / 2
        out_ratio = selected_inner / max(0.001, rog)
        out = -trans_out * out_ratio                # meshed gear counter-rotates
        gen = out                                   # generator is coupled to that gear
        # flywheel is on the core: couples when the main clutch is engaged,
        # otherwise FREE-SPINS with stored energy (slow bearing decay)
        if c1 > 0.05:
            self.fly_rpm_v += (core - self.fly_rpm_v) * min(1.0, dt * 0.4)
        else:
            self.fly_rpm_v *= max(0.0, 1.0 - dt * 0.05)
            core = max(core, self.fly_rpm_v * 0.15)  # flywheel can reduce stall

        self.kin = dict(piston=self.piston_rpm, core=core, gen=gen, out=out,
                        trans_out=trans_out, ratio=eff, gear=gear + 1, ngear=nrings,
                        gr=gr, slip1=(1.0 - c1) * 100.0,
                        bind_slip=(1.0 - bind) * 100.0,
                        fly=self.fly_rpm_v, ep=ep, c1=c1, bind=bind,
                        combust_min=self.piston_rpm * combustion_count(self._eff_comb),
                        super=sc, boost=boost)

        # heat-recovery + regen spinners (driven by last frame's thermal state)
        steam_bar = self.therm.get("steam_bar", 0.0)
        steam_spin = steam_bar * 190.0                    # visual rpm from PSI
        if drive:
            wheel_rpm = abs(self.world.v) / (2 * math.pi * VEH["wheel_radius_m"]) * 60.0
        else:
            wheel_rpm = 70.0 + core * 0.02                # gentle preview roll
        pedal_spin = 220.0 if getattr(self, "pedal_on", False) else 0.0
        # passive drag-air capture turbine spins with airflow; parked windmill spins
        # from wind only when parked/crawling (retracted, still, while driving fast).
        if drive:
            dragturb_spin = wheel_rpm * 2.4 + 180.0 * ep
            parked = abs(self.world.v) <= WINDMILL["park_mph"] / 2.23694
            windmill_spin = 520.0 * clamp(WIND) if parked else 0.0
        else:
            dragturb_spin = 260.0
            windmill_spin = 240.0
        rpm = {"piston": self.piston_rpm, "core": core, "fly": self.fly_rpm_v,
               "gen": gen, "out": out, "super": core * (2.2 + 2.0 * sc),
               "steam": steam_spin, "recov2": core * 1.8 + (300.0 if ep > 0.02 else 0.0),
               "wheel": wheel_rpm, "pedal": pedal_spin, "static": 0.0,
               "dragturb": dragturb_spin, "windmill": windmill_spin}
        for k in range(nrings):
            if k <= gear:
                ring_eff = 1.0 + (ratios[k] - 1.0) * bind
                rpm["trans%d" % k] = core * ring_eff
            else:
                rpm["trans%d" % k] = core * 0.08    # outer rings are not engaged
        ts = 1.0 if drive else (0.0 if self.paused else self.speeds[self.speed_idx])
        for grp, r in rpm.items():
            self.group_angle[grp] = (self.group_angle.get(grp, 0.0)
                                     + (r / 60.0 * 2 * math.pi) * dt * ts * VIEW_ROT_SCALE)
        self.group_angle["default"] = self.group_angle["piston"]
        self.engine_angle = self.group_angle["piston"]

    def _thermal(self, dt, drive):
        """Engine heat + the CLOSED-LOOP steam regeneration.

        Two thermal nodes: the BLOCK (combustion + friction heat) and the compact
        BOILER (working fluid). A geometric heat SHIELD directs block heat into the
        boiler; a chosen FLUID and a DUAL-CHAMBER MULTI-STAGE (compound) expander
        milk the vapour pressure into electricity, reusing it internally until weak.
        Temperature is regulated by releasing pressure (last-resort conditioner is
        the backstop). Returns the kW of recovered electricity (steam + 2nd-stage).
        """
        k = self.kin
        if drive:
            flow = 0.75
            comb_active = 1.0 if self.pt.engine_on else 0.0
            hyd_cmd = max(k["c1"], k["bind"])
            # Clutch slip heat only exists while the range-extender is ACTUALLY driving
            # through the clutch. With the engine OFF the rotary ring is stopped and the
            # clutch disengaged, so there is no slip and no friction heat to conjure --
            # otherwise the model would create heat (then free steam power) from nothing.
            slip_load = comb_active * (0.30 if abs(self.world.v) > 1 else 0.0)
        else:
            flow = self.coolant_flow
            comb_active = 1.0 if k["ep"] > 0.02 else 0.0
            hyd_cmd = self.hyd_pressure
            slip_load = 1.0

        svc = FLUIDS[self.fluid]["service_kwh"]
        self.fluid_life = clamp(1.0 - self.fluid_kwh / max(1e-6, svc))
        self.phys = solve_engine_physics(
            k, self._eff_comb, self.temp_c, flow, hyd_cmd,
            firing=bool(comb_active), slip_load=slip_load,
            n_pistons=self.renderer.n_pistons, boiler_c=self.boiler_c,
            fluid=self.fluid, fluid_health=self.fluid_life)
        heat_in = self.phys["heat_in_block"]
        cool = self.phys["cool_kw"]
        shield_flow = self.phys["shield_flow_kw"]
        heat_boiler = self.phys["heat_in_boiler"]
        steam_draw = self.phys["steam_draw_kw"]

        # BLOCK: combustion+friction in, minus what the shield/vent/condition remove.
        self.temp_c += (heat_in - cool) / PHYS["thermal_mass_kj_c"] * dt
        self.temp_c = max(THERM["ambient_c"], min(PHYS["max_temp_c"], self.temp_c))
        # BOILER (working fluid): directed + exhaust heat in, steam draw + leak out.
        # It holds heat (small mass) so steam keeps flowing after the engine stops;
        # when the block cools the shield keeps feeding it -> sustained trickle.
        # A PHASE-CHANGE MATERIAL (PCM) bank is coupled to the boiler: it absorbs
        # excess heat while firing (pcm_flow +) and gives it back slowly once the
        # boiler cools (pcm_flow -), stretching post-shutdown steam for minutes.
        boiler_leak = THERM["boiler_leak_kw_c"] * max(0.0, self.boiler_c - THERM["ambient_c"])
        pcm_flow = THERM["pcm_couple_kw_c"] * (self.boiler_c - self.pcm_c)   # + boiler->PCM
        # CONCENTRATING SOLAR-TO-BOILER: the windshield lens dumps free sun-heat into
        # the fluid, so the steam loop makes electricity from sunshine even engine-off.
        self.solar_boiler_kw = solar_boiler_kw(self.boiler_c)
        self.boiler_c += (heat_boiler + self.solar_boiler_kw - steam_draw
                          - boiler_leak - pcm_flow) / THERM["boiler_mass_kj_c"] * dt
        self.boiler_c = max(THERM["ambient_c"], min(PHYS["max_temp_c"], self.boiler_c))
        pcm_leak = THERM["pcm_leak_kw_c"] * max(0.0, self.pcm_c - THERM["ambient_c"])
        self.pcm_c += (pcm_flow - pcm_leak) / THERM["pcm_kj_c"] * dt
        self.pcm_c = max(THERM["ambient_c"], min(PHYS["max_temp_c"], self.pcm_c))

        self.phys["temp_c"] = self.temp_c
        self.phys["boiler_c"] = self.boiler_c
        if self.temp_c >= PHYS["shutdown_c"]:
            self.phys["shutdown"] = True
            self.phys["status"] = "SHUTDOWN"
        elif self.temp_c >= PHYS["overheat_c"]:
            self.phys["overheat"] = True
            self.phys["status"] = "OVERHEAT"

        self.therm = {"temp": self.temp_c, "heat_kw": heat_in, "cool_kw": cool,
                      "orc_kw": self.phys["orc_kw"], "main_kw": self.phys["gen_kw"],
                      "total_kw": self.phys["gen_kw"] + self.phys["orc_kw"],
                      "steam_kw": self.phys["steam_kw"], "steam_bar": self.phys["steam_bar"],
                      "boiler_c": self.boiler_c, "exh2_kw": self.phys["exh_stage2_kw"],
                      "pcm_c": self.pcm_c}
        return self.phys["orc_kw"]

    # ---------------------------------------------------------------- update
    def update(self, dt):
        self._frame += 1
        self._rend().tick(dt)                  # tick the active 3D renderer (engine or car)
        drive = (self.mode == self.MODE_DRIVE)
        self.pt.block_temp_c = self.temp_c     # heat cutoff for the engine-off logic
        if drive:
            grade, v_mph = self.world.update(dt, self.throttle, self.brake,
                                             self.pt, self.force_grade,
                                             combustions=self.combustions)
            self.cur_grade = grade
            self.cur_mph = abs(v_mph)
        else:
            self.cur_grade = 0.0
            self.cur_mph = 0.0
        # passenger pedal-assist ramps in/out; adds free human trickle power
        self.pedal_engage += ((1.0 if self.pedal_on else 0.0) - self.pedal_engage) \
            * min(1.0, dt * PEDAL_RAMP)
        pedal_kw = DIMS["seats"] * PEDAL_WATTS_PER_SEAT / 1000.0 * self.pedal_engage
        # pop-out pedals EXTEND only when engaged AND actually helping; otherwise
        # (parked, or too fast for them to matter) they RETRACT into the frame.
        helping = (not drive) or pedals_help(self.cur_mph, DIMS["seats"])
        tgt_ext = 1.0 if (self.pedal_on and helping) else 0.0
        self.pedal_extend += (tgt_ext - self.pedal_extend) * min(1.0, dt * 3.0)
        self.renderer.pedal_extend = self.pedal_extend
        # MPG strategy: the corrected operating-point sweep found the best net point
        # at low RPM with all 8 chambers per rev: same useful generator power, lower
        # bearing speed, and bounded same-second heat recovery.
        self._eff_comb = self.combustions
        self._kinematics(dt, drive)
        orc = self._thermal(dt, drive)
        if drive:
            if self.phys.get("shutdown"):
                self.pt.engine_on = False
                self.pt.force_timer = 0.0
            # working-fluid service life is consumed by recovered energy (wear is
            # faster if the boiler runs hotter than the fluid tolerates)
            wear = 1.0 + 0.03 * max(0.0, self.boiler_c - FLUIDS[self.fluid]["heat_ok_c"])
            self.fluid_kwh += max(0.0, orc) * (dt / 3600.0) * wear
            # Multi-layer AMBIENT harvest (solar + suspension + TENG + tyre + piezo):
            # always on, fuel-free, and it charges even coasting with the engine off.
            self.ambient_kw = ambient_harvest_w(self.cur_mph) / 1000.0
            self.harvest_wh += self.ambient_kw * 1000.0 * (dt / 3600.0)
            # NEW fuel-free paths: thermoelectric (TEG) straight off the hot block,
            # dynamic electromagnetic-suspension bump harvest, and regenerative
            # steering. All routed straight into storage; all raise MPG.
            self.teg_kw = teg_harvest_kw(self.temp_c)
            self.susp_kw = suspension_bump_w(self.cur_mph, self.cur_grade) / 1000.0
            self.steer_kw = steering_regen_kw(self.cur_mph, self.world.steer)
            self.solar_roof_kw = AMBIENT_HARVEST_W["solar_roof"] * SUN / 1000.0
            # PASSIVE DRAG-AIR CAPTURE (moving) + PARKED WINDMILL (stopped): recover
            # otherwise-wasted drag energy, and trickle-charge from wind when parked.
            self.drag_kw = drag_capture_kw(self.cur_mph)
            self.wind_kw = windmill_kw(self.cur_mph)
            # INERTIAL PENDULUM RIM: harvest the body's own rocking from accel/brake,
            # cornering, and incline changes (effective-gravity swings of the bobs).
            long_g = (self.world.v - self._prev_v) / max(1e-6, dt) / VEH["g"]
            lat_g = self.world.steer * self.world.v * 0.16 / VEH["g"]
            pitch_rate = (self.cur_grade - self._prev_grade) / max(1e-6, dt)
            self._prev_v = self.world.v
            self._prev_grade = self.cur_grade
            # only harvests while DEPLOYED; retracts (harvest -> 0) as speed rises so
            # it never adds net drag at cruise.
            self.pend_deploy = pendulum_deploy(self.cur_mph)
            self.pend_kw = (pendulum_harvest_w(long_g, lat_g, pitch_rate) / 1000.0
                            * self.pend_deploy)
            extra_kw = (self.teg_kw + self.susp_kw + self.steer_kw
                        + self.drag_kw + self.wind_kw + self.pend_kw)
            self.extra_wh += extra_kw * 1000.0 * (dt / 3600.0)
            self.cd_eff = self.world.cd_eff
            self.future_grade = self.world.future_grade
            self.soc_target = self.pt.soc_target
            # Physics: main generator + recovered heat + pedal + ambient + new paths.
            charge_kw = (self.therm["main_kw"] + orc + pedal_kw + self.ambient_kw
                         + extra_kw)
            # NET traction against every charge source in ONE routing pass: energy used
            # the same instant for traction bypasses the storage round-trip entirely;
            # only the true surplus is buffered (or deficit pulled). Routing synergy.
            total_kw = self.pt._route_net_kw + charge_kw
            self.pt.route_storage(total_kw * (dt / 3600.0), dt)
            self.pedal_wh += pedal_kw * 1000.0 * (dt / 3600.0)
            self.pt.flow["pedal"] = pedal_kw
            self.pt.flow["ambient"] = self.ambient_kw
            # HV electrical bus: the rail carries the traction draw (or regen charge);
            # compute the live pack voltage / bus current / I^2R + inverter loss.
            self.elec = solve_electrical(self.pt.flow.get("trac", 0.0)
                                         + self.pt.flow.get("regen", 0.0))
            if self.phys.get("fuel_g_s", 0.0) > 0.0:
                self.pt.fuel_gal += (self.phys["fuel_g_s"] / 1000.0) * dt \
                    / PHYS["gasoline_kg_per_gal"]
            if self.pt.fuel_gal > 1e-6:
                self.pt.last_mpg = self.pt.miles / self.pt.fuel_gal
            else:
                self.pt.last_mpg = 9999.0
            self.pt.flow["engine"] = self.therm["main_kw"]
            self.pt.flow["orc"] = orc
            self._compute_firing(self.pt.engine_on, self._eff_comb, self.fire_mode)
        else:
            self._compute_firing(self.engine_power > 0.02, self._eff_comb, self.fire_mode)

    # ----------------------------------------------------------------- draw
    def draw(self):
        self.screen.blit(self.bg, (0, 0))
        if self.mode == self.MODE_PREVIEW:
            self.draw_preview()
        elif self.mode == self.MODE_CAR:
            self.draw_carview()
        else:
            self.draw_drive()
        if self.show_help:
            self.draw_help()
        if self.show_info:
            self.draw_info()
        if self.show_mpg:
            self.draw_mpg_chart()
        self.draw_topbar()
        pygame.display.flip()

    def draw_topbar(self):
        panel(self.screen, 0, 0, self.W, 34, alpha=190)
        vname = {"full": "FULL", "exploded": "EXPLODED", "assembly": "ASSEMBLY"}
        if self.mode == self.MODE_PREVIEW:
            cut = " +SECTION-CUT" if self.renderer.section else ""
            mode = "ENGINE PREVIEW [%s%s]" % (vname.get(self.renderer.view, "FULL"), cut)
        elif self.mode == self.MODE_CAR:
            cut = " +SECTION-CUT" if self.car.section else ""
            mode = "CAR VIEW [%s%s]" % (vname.get(self.car.view, "FULL"), cut)
        else:
            mode = "DRIVE  (simulation)"
        self.screen.blit(self.fbig.render("GmansRun V1.17", True, C_ACCENT), (12, 2))
        t = self.font.render("MODE: %s   [TAB] engine>car>drive  [M] MPG  [I] info  [H] help" % mode,
                             True, C_TEXT)
        self.screen.blit(t, (self.W - t.get_width() - 12, 9))

    # ---- preview ----
    def _heat_norm(self):
        return max(0.0, min(1.0, (self.temp_c - THERM["ambient_c"])
                            / (PHYS["overheat_c"] - THERM["ambient_c"])))

    def draw_preview(self):
        rect = pygame.Rect(0, 34, self.W, self.H - 34)
        self.renderer.render(self.screen, rect, self.group_angle, self.pt.firing_glow,
                             mouse_pos=pygame.mouse.get_pos(),
                             show_labels=self.show_labels, label_font=self.fsmall,
                             interactive=True, injection=self._injection,
                             heat=self._heat_norm(),
                             selector_radius=output_gear_mesh_radius(self.kin["gear"] - 1))
        self.draw_view_tabs()
        self.draw_spec_card()
        self.draw_bench_panel()
        self.draw_config_panel()
        self.draw_power_path()

    def draw_carview(self):
        """The separate WHOLE-CAR 3D view: the entire vehicle to scale with the same
        full / exploded / assembly + section-cut, hover-to-inspect and orbit controls
        as the engine preview, but rendered from the car renderer."""
        rect = pygame.Rect(0, 34, self.W, self.H - 34)
        self.car.render(self.screen, rect, self.group_angle, self.pt.firing_glow,
                        mouse_pos=pygame.mouse.get_pos(),
                        show_labels=self.show_labels, label_font=self.fsmall,
                        interactive=True, heat=0.0)
        self.draw_view_tabs()
        self.draw_spec_card()
        # whole-car dimensional readout (real-life scale)
        px, pw = 12, 300
        py = 158 if self.show_help else 132
        panel(self.screen, px, py, pw, 150)
        self.screen.blit(self.fsmall.render("WHOLE-CAR MODEL  (real-life scale)",
                         True, C_ACCENT), (px + 10, py + 8))
        fa = CAR["width_mm"] / 1000.0 * CAR["height_mm"] / 1000.0 * CAR["frontal_fill"]
        info = [
            ("Size L/W/H", "%.0f/%.0f/%.0f mm" % (
                CAR["length_mm"], CAR["width_mm"], CAR["height_mm"])),
            ("Wheelbase/track", "%.0f / %.0f mm" % (CAR["wheelbase_mm"], CAR["track_mm"])),
            ("Curb mass", "%.0f kg" % VEH["curb_mass_kg"]),
            ("Cd / area", "%.2f / %.2f m2" % (VEH["Cd"], fa)),
            ("Crr (mag brg)", "%.4f" % effective_crr()),
            ("Wheel drives", "4 rail/disc modules"),
            ("Parts", "%d  (hover to inspect)" % len(self.car.parts)),
        ]
        for i, (lab, val) in enumerate(info):
            ry = py + 28 + i * 17
            self.screen.blit(self.fsmall.render(lab, True, C_TEXT_DIM), (px + 10, ry))
            vimg = self.fsmall.render(val, True, C_TEXT)
            self.screen.blit(vimg, (px + pw - 10 - vimg.get_width(), ry))
        hint = ("1 FULL  2 EXPLODED  3 ASSEMBLY  4/X SECTION   drag orbit | wheel zoom | "
                "right/middle pan | L (or LABELS chip) = names | [TAB] engine>car>drive")
        self.screen.blit(self.fsmall.render(hint, True, C_TEXT_DIM),
                         (14, self.H - 26))

    def _bench_specs(self):
        nrings = DIMS["trans_rings"]
        return [
            ("Engine Power", "engine_power", 0.0, 1.0, False),
            ("Hydraulic Pressure", "hyd_pressure", 0.0, 1.0, False),
            ("Main Wet Clutch (grip)", "clutch_main", 0.0, 1.0, False),
            ("Trans Wet Clutch (bind)", "trans_bind_cmd", 0.0, 1.0, False),
            ("Coolant Flow", "coolant_flow", 0.0, 1.0, False),
            ("Set Gear (ring shift)", "gear_sel", 0, nrings - 1, True),
        ]

    def _bench_layout(self):
        # sit clear of the help panel (which occupies the top-left when shown)
        px, pw = 12, 278
        py = 158 if self.show_help else 132
        rects = []
        y = py + 40
        for _ in self._bench_specs():
            rects.append(pygame.Rect(px + 14, y, pw - 28, 8))
            y += 36
        return px, py, pw, y, rects

    def draw_bench_panel(self):
        specs = self._bench_specs()
        px, py, pw, yend, rects = self._bench_layout()
        panel_h = (yend - py) + 336
        panel(self.screen, px, py, pw, panel_h)
        self.screen.blit(self.fsmall.render("ENGINE TEST BENCH  (drag sliders)",
                                            True, C_ACCENT), (px + 10, py + 8))
        for (label, attr, lo, hi, disc), r in zip(specs, rects):
            val = getattr(self, attr)
            frac = (val - lo) / (hi - lo) if hi > lo else 0.0
            frac = max(0.0, min(1.0, frac))
            valtxt = ("%d / %d" % (int(val) + 1, int(hi) + 1) if disc
                      else "%3.0f%%" % (val * 100))
            self.screen.blit(self.fsmall.render(label, True, C_TEXT), (r.x, r.y - 14))
            vt = self.fsmall.render(valtxt, True, C_GOOD)
            self.screen.blit(vt, (r.right - vt.get_width(), r.y - 14))
            pygame.draw.rect(self.screen, C_PANEL_HI, r, border_radius=4)
            pygame.draw.rect(self.screen, C_ACCENT, (r.x, r.y, int(r.w * frac), r.h),
                             border_radius=4)
            kx = r.x + int(r.w * frac)
            pygame.draw.circle(self.screen, C_TEXT, (kx, r.y + r.h // 2), 6)
            pygame.draw.circle(self.screen, C_ACCENT, (kx, r.y + r.h // 2), 6, 2)
        # live readouts (kinematics + thermal / recovery)
        k, t, p = self.kin, self.therm, self.phys
        if p["shutdown"]:
            tcol = C_BAD
        elif p["overheat"]:
            tcol = C_WARN
        elif t["temp"] > THERM["steam_c"]:
            tcol = C_ACCENT
        else:
            tcol = C_GOOD
        rows = [
            ("Piston", "%5.0f rpm" % k["piston"], C_RING_HOT),
            ("Core", "%5.0f  clutch slip %2.0f%%" % (k["core"], k["slip1"]), C_TEXT),
            ("Ratio (bind)", "%.2f:1  slip %2.0f%%" % (k["ratio"], k["bind_slip"]), C_TEXT),
            ("TransOut / Gen", "%5.0f / %5.0f" % (k["trans_out"], abs(k["gen"])), C_GOOD),
            ("Gear / Fly", "%d of %d  |  %4.0f" % (k["gear"], k["ngear"], k["fly"]), C_TEXT),
            ("Comb/min", "%7.0f  (%d/rev)" % (k["combust_min"], combustion_count(self.combustions)), C_RING_HOT),
            ("Press / PR", "%3.0f kPa  %.2fx" % (p["pressure_kpa"], p["pressure_ratio"]), C_ACCENT),
            ("Air / Fuel", "%4.1f / %4.2f g/s" % (p["air_g_s"], p["fuel_g_s"]), C_TEXT),
            ("Events / Chamber", "%4.0f/s  %3.0f cc" % (p["events_s"], p["chamber_cc"]), C_TEXT_DIM),
            ("Hyd clamp", "%3.0f bar %3.0f/%3.0f kN" % (
                p["hyd_bar"], p["main_clamp_kn"], p["ring_clamp_kn"]), C_WARN),
            ("Super / Boost", "%3.0f%%  %.2fx" % (k["super"] * 100, k["boost"]), C_ACCENT),
            ("Friction", "%3.0f kW  bearing %.1f" % (p["friction_kw"], p["bearing_kw"]), C_WARN),
            ("Block / Boiler", "%3.0f / %3.0f C" % (t["temp"], t.get("boiler_c", 0.0)), tcol),
            ("Fluid / PSI", "%s %4.1f/%2.0fbar" % (
                p.get("fluid", "-"), p["steam_bar"], p.get("steam_max_bar", 0.0)), C_ACCENT),
            ("Fluid life/heat", "%3.0f%%  ok<%3.0fC" % (
                self.fluid_life * 100, FLUIDS[self.fluid]["heat_ok_c"]),
             C_GOOD if self.fluid_life > 0.15 else C_BAD),
            ("Shield -> Boil", "%2.0f -> %2.0f kW  x%d" % (
                p.get("shield_flow_kw", 0.0), p["steam_draw_kw"], THERM["compound_stages"]), C_TEXT_DIM),
            ("Recover st+ex2", "%2.0f + %2.0f kW" % (p["steam_kw"], p["exh_stage2_kw"]), C_GOOD),
            ("Elec main+recov", "%2.0f + %2.0f kW" % (t["main_kw"], t["orc_kw"]), C_GOOD),
            ("Ambient harvest", "%3.0f W  solar+susp+TENG" % ambient_harvest_w(self.cur_mph), C_ACCENT),
            ("Pedal assist", "%s  %3.0f W" % (
                "ON" if self.pedal_on else "off",
                DIMS["seats"] * PEDAL_WATTS_PER_SEAT * self.pedal_engage), C_GOOD if self.pedal_on else C_TEXT_DIM),
            ("Thermal State", "%s  derate %3.0f%%" % (p["status"], p["derate"] * 100), tcol),
        ]
        oy = yend + 4
        for label, val, col in rows:
            self.screen.blit(self.fsmall.render(label, True, C_TEXT_DIM), (px + 12, oy))
            v = self.fsmall.render(val, True, col)
            self.screen.blit(v, (px + pw - 12 - v.get_width(), oy))
            oy += 15

    def _bench_hit(self, pos):
        """If the mouse is on a bench slider, return its index, else None."""
        _, _, _, _, rects = self._bench_layout()
        for i, r in enumerate(rects):
            if r.inflate(16, 20).collidepoint(pos):
                return i
        return None

    def _bench_set_from_x(self, idx, mx):
        _, _, _, _, rects = self._bench_layout()
        label, attr, lo, hi, disc = self._bench_specs()[idx]
        r = rects[idx]
        frac = max(0.0, min(1.0, (mx - r.x) / r.w))
        val = lo + frac * (hi - lo)
        setattr(self, attr, int(round(val)) if disc else val)

    def _config_geom(self):
        return self.W - 332, self.H - 252, 320, 240

    def _combust_slider_rect(self):
        x, y, w, h = self._config_geom()
        return pygame.Rect(x + 14, y + 204, w - 120, 8)

    def _combust_hit(self, pos):
        return self._combust_slider_rect().inflate(16, 22).collidepoint(pos)

    def _combust_set_from_x(self, mx):
        r = self._combust_slider_rect()
        frac = max(0.0, min(1.0, (mx - r.x) / r.w))
        self.combustions = int(round(1 + frac * (DIMS["chambers"] - 1)))

    def draw_config_panel(self):
        n = self.renderer.n_pistons
        speed = "PAUSED" if self.paused else "%.2fx" % self.speeds[self.speed_idx]
        scol = C_WARN if self.paused else C_GOOD
        x, y, w, h = self._config_geom()
        panel(self.screen, x, y, w, h)
        self.screen.blit(self.fsmall.render("LIVE ENGINE CONTROLS", True, C_ACCENT), (x + 10, y + 8))
        fl = FLUIDS[self.fluid]
        freezes = fl["freeze_c"] >= -5.0
        frz_note = "FREEZES!" if freezes else "no-freeze"
        frz_col = C_BAD if freezes else C_GOOD
        rows = [
            ("PISTONS", "%d  (1-10)" % n, "[ ]", C_TEXT),
            ("COMBUSTIONS / REV", "%d  (1-8)" % self.combustions, ",  .", C_RING_HOT),
            ("COMBUSTIONS / MIN", "%7.0f" % self.kin["combust_min"], "rpm x count", C_ACCENT),
            ("BOILER FLUID", "%s" % self.fluid, "F cycle", C_ACCENT),
            ("FLUID boil/freeze", "%2.0fC / %+.0fC" % (fl["boil_c"], fl["freeze_c"]),
             frz_note, frz_col),
            ("SUPERCHARGER", "%3.0f%% auto" % (self.kin["super"] * 100), "engine on", C_GOOD),
            ("VIEW SPEED", speed, "P  - =", scol),
        ]
        oy = y + 30
        for label, val, keys, col in rows:
            self.screen.blit(self.fsmall.render(label, True, C_TEXT_DIM), (x + 10, oy))
            self.screen.blit(self.fsmall.render(val, True, col), (x + 152, oy))
            self.screen.blit(self.fsmall.render(keys, True, C_TEXT_DIM), (x + 248, oy))
            oy += 24
        # draggable COMBUSTIONS/REV bar (1..8) -- the on-screen combustion control
        r = self._combust_slider_rect()
        self.screen.blit(self.fsmall.render("COMBUSTIONS / REV  (drag 1-8)", True, C_TEXT_DIM),
                         (r.x, r.y - 14))
        frac = (self.combustions - 1) / float(max(1, DIMS["chambers"] - 1))
        pygame.draw.rect(self.screen, C_PANEL_HI, r, border_radius=4)
        pygame.draw.rect(self.screen, C_RING_HOT, (r.x, r.y, int(r.w * frac), r.h), border_radius=4)
        kx = r.x + int(r.w * frac)
        pygame.draw.circle(self.screen, C_TEXT, (kx, r.y + r.h // 2), 6)
        pygame.draw.circle(self.screen, C_RING_HOT, (kx, r.y + r.h // 2), 6, 2)
        vt = self.fsmall.render("%d / %d" % (self.combustions, DIMS["chambers"]), True, C_RING_HOT)
        self.screen.blit(vt, (r.right + 14, r.y - 5))

    def draw_view_tabs(self):
        r = self._rend()
        x, y = 12, 42
        tabs = (("1", "FULL", "full"), ("2", "EXPLODED", "exploded"),
                ("3", "ASSEMBLY", "assembly"), ("4", "SECTION-CUT", "section"))
        for key, name, mode in tabs:
            cur = r.section if mode == "section" else (r.view == mode)
            img = self.fsmall.render("%s %s" % (key, name), True,
                                     C_TEXT if cur else C_TEXT_DIM)
            w = img.get_width() + 18
            panel(self.screen, x, y, w, 22, alpha=225 if cur else 150)
            if cur:
                pygame.draw.rect(self.screen, C_ACCENT, (x, y, w, 22), 1, border_radius=6)
            self.screen.blit(img, (x + 9, y + 4))
            x += w + 6
        # LABELS toggle chip -- a visible, clickable button for the L hotkey so the
        # part-name callouts can be cleared for an unobstructed view.
        on = self.show_labels
        ltxt = "L LABELS: %s" % ("ON" if on else "OFF")
        limg = self.fsmall.render(ltxt, True, C_TEXT if on else C_TEXT_DIM)
        lw = limg.get_width() + 18
        panel(self.screen, x, y, lw, 22, alpha=225 if on else 150)
        pygame.draw.rect(self.screen, C_ACCENT if on else C_TEXT_DIM,
                         (x, y, lw, 22), 1, border_radius=6)
        self.screen.blit(limg, (x + 9, y + 4))
        self._labels_chip = pygame.Rect(x, y, lw, 22)      # click target for the toggle
        x += lw + 6
        if r.view == "assembly":
            prog = "Assembled %d / %d   (click or N = place next, B = back, F = all)" % (
                r.assembled, len(r.parts))
            self.screen.blit(self.fsmall.render(prog, True, C_WARN), (x + 6, y + 4))

    def draw_spec_card(self):
        r = self._rend()
        part = r.active_part()
        placing = None
        if part is None and r.view == "assembly":
            placing = r.placing_part()
            part = placing
        w, x, y = 372, self.W - 384, 74
        if part is None:
            panel(self.screen, x, y, w, 60)
            self.screen.blit(self.fsmall.render("INSPECTOR", True, C_ACCENT), (x + 10, y + 8))
            self.screen.blit(self.fsmall.render("Hover a part to read its spec; click to pin.",
                                                True, C_TEXT_DIM), (x + 10, y + 30))
            return
        wrapped = []
        for ln in part.specs:
            wrapped.extend(wrap_text(self.fsmall, ln, w - 22))
        h = 56 + len(wrapped) * 18
        panel(self.screen, x, y, w, h)
        if r.selected is not None and placing is None:
            tag = "PINNED PART"
        elif placing is not None:
            tag = "NEXT TO PLACE"
        else:
            tag = "INSPECTOR"
        self.screen.blit(self.fsmall.render(tag, True, C_ACCENT), (x + 10, y + 8))
        name = wrap_text(self.font, part.name, w - 22)[0]
        self.screen.blit(self.font.render(name, True, C_TEXT), (x + 10, y + 26))
        oy = y + 52
        for ln in wrapped:
            self.screen.blit(self.fsmall.render(ln, True, C_TEXT_DIM), (x + 10, oy))
            oy += 18

    def draw_power_path(self):
        firing_now = int((self.pt.firing_glow > 0.4).sum())
        ccol = C_RING_HOT if firing_now else C_TEXT_DIM
        self.screen.blit(self.fsmall.render(
            "Combustion: %d chamber glow  mode %d/rev   %.0f combustions/min   Piston %.0f rpm   Generator %.0f rpm" % (
                firing_now, combustion_count(self.combustions), self.kin["combust_min"],
                self.kin["piston"], abs(self.kin["out"])), True, ccol), (14, self.H - 124))
        lines = [
            "POWER PATH  (fuel -> charge):",
            "turbo + auto supercharger -> sealed chamber -> 8-chamber ring -> multi-plate main clutch",
            "-> gear wet-clutch rings -> receival gear -> MAIN generator -> supercaps/battery",
            "HEAT RECOVER: turbo -> 2nd turbine -> cooling ring -> heat shield -> boiler (low-boil fluid)",
            "-> dual-chamber compound expander (reuse PSI x3, cool by VENTING) || REGEN: wheel-gens",
        ]
        panel(self.screen, 12, self.H - 108, 838, 96)
        for i, ln in enumerate(lines):
            self.screen.blit(self.fsmall.render(ln, True, C_ACCENT if i == 0 else C_TEXT),
                             (22, self.H - 102 + i * 17))

    # ---- drive ----
    def draw_drive(self):
        self.draw_road()
        self.draw_car3d()
        self.draw_drive_hud()
        iw, ih = 360, 300
        ir = pygame.Rect(self.W - iw - 12, 44, iw, ih)
        panel(self.screen, ir.x, ir.y, ir.w, ir.h, alpha=180)
        self.screen.blit(self.fsmall.render("LIVE ENGINE", True, C_ACCENT),
                         (ir.x + 8, ir.y + 4))
        # the spinning full-engine inset is the most expensive thing in the drive
        # loop, so render it into a cached surface every OTHER frame (imperceptible
        # at 60 fps) and blit the cache in between -- protects the frame rate.
        ew, eh = iw, ih - 16
        if self._inset_cache is None or self._inset_cache.get_size() != (ew, eh):
            self._inset_cache = pygame.Surface((ew, eh), pygame.SRCALPHA)
        if self._frame % 2 == 0:
            self._inset_cache.fill((0, 0, 0, 0))
            self.renderer.render(self._inset_cache, pygame.Rect(0, 0, ew, eh),
                                 self.group_angle, self.pt.firing_glow,
                                 show_labels=False, view_override="full",
                                 injection=self._injection, heat=self._heat_norm(),
                                 selector_radius=output_gear_mesh_radius(self.kin["gear"] - 1))
        self.screen.blit(self._inset_cache, (ir.x, ir.y + 16))

    def draw_road(self):
        horizon = int(self.H * 0.42) - int(self.cur_grade * 600)
        horizon = max(120, min(self.H - 200, horizon))
        self.screen.blit(self._sky, (0, 34), area=pygame.Rect(0, 0, self.W, horizon - 34))
        pygame.draw.rect(self.screen, C_GRASS, (0, horizon, self.W, self.H - horizon))
        cxv = self.W // 2 + int(self.world.steer * 60)
        road_w_bot = self.W * 0.8
        pts = [(cxv - 40, horizon), (cxv + 40, horizon),
               (self.W / 2 + road_w_bot / 2, self.H),
               (self.W / 2 - road_w_bot / 2, self.H)]
        pygame.draw.polygon(self.screen, C_ROAD, pts)
        phase = (self.world.dist_m * 0.5) % 1.0
        for k in range(0, 24):
            t0 = (k + phase) / 24.0
            t1 = (k + 0.5 + phase) / 24.0
            if t0 >= 1 or t1 >= 1:
                continue
            y0 = horizon + (self.H - horizon) * t0 ** 2
            y1 = horizon + (self.H - horizon) * t1 ** 2
            w0 = 1 + 6 * t0 ** 2
            xx = cxv + (self.W / 2 - cxv) * (t0 ** 2)
            pygame.draw.line(self.screen, C_ROAD_LINE, (xx, y0), (xx, y1), int(1 + w0))

    def draw_car3d(self):
        """Render the REAL 3D vehicle model (same model as CAR VIEW) from a fixed
        3rd-person chase camera, sitting on the road. The wheels spin at the true
        road speed via the shared 'wheel' group angle; the view banks + slides with
        the steering, and brake/reverse lights + the live aero badge overlay it."""
        steer = self.world.steer
        # chase camera: behind + slightly above the car, looking down the road; it
        # yaws a touch INTO the corner, the car slides across the road with steer,
        # pulls back with speed and pitches with the road grade.
        self.drivecar.az = math.pi + 0.22 - steer * 0.12
        self.drivecar.el = -0.16 - clamp(self.cur_grade * 0.6, -0.12, 0.12)
        self.drivecar.dist = 5.0 + min(1.3, self.cur_mph / 90.0)
        cw, ch = 760, 470
        cxr = int(self.W / 2 - cw / 2 + steer * 48)
        rect = pygame.Rect(cxr, self.H - ch - 4, cw, ch)
        self.drivecar.pan = np.array([0.0, 40.0])         # seat the car low in the frame
        # the rear light bar + reverse lights are REAL meshes on the model; they
        # brighten via the render glow params (no 2D overlay -> they track the body).
        brake_glow = min(1.0, self.brake * 1.4) if self.brake > 0.05 else 0.0
        reverse_glow = 1.0 if (self.world.gear < 0 and abs(self.world.v) > 0.2) else 0.0
        self.drivecar.render(self.screen, rect, self.group_angle, self.pt.firing_glow,
                             show_labels=False, view_override="full",
                             injection=None, heat=0.0,
                             brake_glow=brake_glow, reverse_glow=reverse_glow)
        ccx = rect.centerx
        # live aero badge (morphed + hydrophobic Cd) above the car
        badge = "AI-AERO  Cd %.3f  A %.2fm2  RH %2.0f%%" % (
            self.cd_eff, VEH["frontal_area_m2"], HUMIDITY * 100)
        b = self.fsmall.render(badge, True, (150, 200, 235))
        self.screen.blit(b, (ccx - b.get_width() // 2, rect.top + 6))
        # compact fuel-free effect indicators down the right edge of the car frame
        tags = []
        if self.cd_eff < VEH["Cd"] - 1e-4:
            tags.append(("AERO MORPH", C_GOOD))
        if self.cur_mph > 20:
            tags.append(("FLOW-THRU DUCT", (110, 190, 240)))
        if HUMIDITY > 0.05:
            tags.append(("HYDROPHOBIC SKIN", (150, 205, 235)))
        if self.pend_deploy > 0.05 and self.pend_kw > 1e-4:
            tags.append(("PENDULUM %3.0fW" % (self.pend_kw * 1000), (150, 205, 235)))
        else:
            tags.append(("PENDULUM RETRACTED", C_TEXT_DIM))
        for i, (txt, col) in enumerate(tags):
            t = self.fsmall.render(txt, True, col)
            self.screen.blit(t, (rect.right - t.get_width() - 10, rect.top + 28 + i * 16))

    def draw_car(self):
        # AI-optimized aerodynamic body: a low, narrow TEARDROP cabin with a long
        # boat-tail and FAIRED (covered) wheels -- drawn instead of a boxy sedan to
        # show the shape that gets Cd ~0.08 and the low road load / high MPG.
        cx = self.W // 2 + int(self.world.steer * 30)
        cy = self.H - 88
        c_body, c_edge, c_glass = (46, 122, 192), (26, 78, 132), (150, 200, 235)
        # faired wheels: full wheels drawn first, body then covers their tops
        for sx in (-70, 70):
            pygame.draw.circle(self.screen, (15, 15, 18), (cx + sx, cy + 34), 15)
            pygame.draw.circle(self.screen, (60, 60, 66), (cx + sx, cy + 34), 6)
        # smooth teardrop body (rounded nose far, tapered boat-tail toward viewer)
        body = [(cx - 46, cy + 30), (cx - 86, cy + 6), (cx - 82, cy - 10),
                (cx - 46, cy - 34), (cx - 18, cy - 44), (cx + 18, cy - 44),
                (cx + 46, cy - 34), (cx + 82, cy - 10), (cx + 86, cy + 6),
                (cx + 46, cy + 30)]
        pygame.draw.polygon(self.screen, c_body, body)
        pygame.draw.polygon(self.screen, c_edge, body, 2)
        # low bubble canopy (teardrop glass)
        canopy = [(cx - 30, cy - 16), (cx - 16, cy - 40), (cx + 16, cy - 40),
                  (cx + 30, cy - 16), (cx + 18, cy - 6), (cx - 18, cy - 6)]
        pygame.draw.polygon(self.screen, c_glass, canopy)
        pygame.draw.line(self.screen, (95, 155, 215), (cx, cy - 42), (cx, cy + 28), 2)
        # ACTIVE MORPHING AERO: deployed boat-tail lip + underbody skirts at cruise
        morph = clamp((VEH["Cd"] - self.cd_eff)
                      / max(1e-6, VEH["Cd"] * (1.0 - AERO["morph_cd_min_frac"])))
        if morph > 0.03:
            ext = int(3 + 9 * morph)
            pygame.draw.polygon(self.screen, (34, 96, 158),
                [(cx - 40, cy + 28), (cx + 40, cy + 28),
                 (cx + 26, cy + 28 + ext), (cx - 26, cy + 28 + ext)])
            for sx in (-70, 70):        # underbody side skirts sealing the wheel wells
                pygame.draw.rect(self.screen, (24, 78, 132),
                                 (cx + sx - 10, cy + 30, 20, 2 + int(4 * morph)))
            self.screen.blit(self.fsmall.render("AERO MORPH", True, C_GOOD),
                             (cx + 52, cy + 22))
        # SUPER-HYDROPHOBIC SKIN: in humid/wet air, rain beads up and rolls straight
        # off the body instead of clinging as a drag-adding film. Drawn as droplets
        # that bead on the canopy and roll off the boat-tail.
        if HUMIDITY > 0.05:
            ph = (pygame.time.get_ticks() / 900.0)
            for i in range(6):
                roll = (ph + i * 0.37) % 1.0
                dx = cx - 26 + i * 10
                dy = int(cy - 34 + roll * 62)          # slides down the body and off
                rad = 3 if i % 2 == 0 else 2
                pygame.draw.circle(self.screen, (150, 205, 235), (dx, dy), rad)
                pygame.draw.circle(self.screen, (235, 248, 255), (dx - 1, dy - 1), 1)
            self.screen.blit(self.fsmall.render("HYDROPHOBIC", True, (150, 205, 235)),
                             (cx - 118, cy - 40))
        # THROUGH-BODY FLOW DUCT: at speed, nose air flows STRAIGHT THROUGH the car and
        # fills the rear wake -- faint blue streaks front->rear + a wake-fill puff.
        if self.cur_mph > 20:
            fcol = (110, 190, 240)
            ph = pygame.time.get_ticks() / 260.0
            for i in range(4):
                sx = cx - 24 + i * 16
                t0 = (ph + i * 0.25) % 1.0
                y0 = cy - 40 + t0 * 78
                pygame.draw.line(self.screen, fcol, (sx, int(y0)),
                                 (sx, int(min(cy + 42, y0 + 13))), 2)
            for rr in (5, 9):                       # wake-fill puff behind the tail
                pygame.draw.circle(self.screen, (90, 150, 200), (cx, cy + 48), rr, 1)
            self.screen.blit(self.fsmall.render("FLOW-THRU DUCT", True, fcol),
                             (cx + 40, cy + 40))
        # INERTIAL PENDULUM RIM: the hanging bumper-skirt weights swing with cornering
        # + accel/brake + incline changes, driving little generators. The skirt
        # RETRACTS flush into the body as speed rises (pend_deploy -> 0).
        pcol = (150, 205, 235)
        dep = self.pend_deploy
        pygame.draw.line(self.screen, (70, 80, 100), (cx - 80, cy + 31), (cx + 80, cy + 31), 2)
        if dep > 0.05:
            act = clamp(self.pend_kw / 0.12)
            sway = self.world.steer * 9 + math.sin(pygame.time.get_ticks() / 170.0) * (2 + 9 * act)
            hang = int(10 * dep)                       # bobs pull up as it retracts
            for i in range(7):
                bx = cx - 66 + i * 22
                tipx = bx + int(sway * dep)
                pygame.draw.line(self.screen, (120, 130, 150),
                                 (bx, cy + 31), (tipx, cy + 31 + hang), 1)
                pygame.draw.circle(self.screen, pcol, (tipx, cy + 33 + hang), 3)
            self.screen.blit(self.fsmall.render("PENDULUM %3.0fW" % (self.pend_kw * 1000),
                             True, pcol), (cx - 122, cy - 24))
        else:
            self.screen.blit(self.fsmall.render("PENDULUM RETRACTED", True, C_TEXT_DIM),
                             (cx - 122, cy - 24))
        # aero badge (shows the LIVE morphed + hydrophobic Cd)
        self.screen.blit(self.fsmall.render("AI-AERO  Cd %.3f  A %.2fm2  RH %2.0f%%" % (
            self.cd_eff, VEH["frontal_area_m2"], HUMIDITY * 100), True, c_glass),
            (cx - 62, cy - 60))
        if self.brake > 0.1:
            for sx in (-58, 58):
                pygame.draw.circle(self.screen, (255, 50, 40), (cx + sx, cy - 2), 5)
        if self.world.gear < 0 and abs(self.world.v) > 0.2:   # reverse lights
            for sx in (-40, 40):
                pygame.draw.circle(self.screen, (240, 240, 230), (cx + sx, cy + 6), 4)

    def draw_drive_hud(self):
        pt = self.pt
        panel(self.screen, 12, 44, 300, 150)
        self.screen.blit(self.fhuge.render("%5.1f" % self.cur_mph, True, C_TEXT), (20, 50))
        self.screen.blit(self.font.render("MPH", True, C_TEXT_DIM), (210, 100))
        # gear indicator (D / R)
        rev = self.world.gear < 0
        gcol = C_BAD if rev else C_GOOD
        self.screen.blit(self.fbig.render("R" if rev else "D", True, gcol), (262, 52))
        self.screen.blit(self.fsmall.render("REV (R)" if rev else "DRIVE (R)", True, C_TEXT_DIM),
                         (235, 92))
        mpg = pt.last_mpg
        mpg_txt = "----" if mpg >= 9999 else "{:,.0f}".format(mpg)
        col = C_GOOD if mpg >= 800 else (C_WARN if mpg >= 300 else C_BAD)
        self.screen.blit(self.fbig.render("%s MPG" % mpg_txt, True, col), (20, 130))
        if self.world.cruise:
            self.screen.blit(self.font.render("CRUISE ON", True, C_GOOD), (160, 165))

        x, y, w = 24, self.H - 150, 260
        bar(self.screen, self.font, x, y, w, 14, pt.soc, C_ACCENT,
            "BATTERY SOC", "%4.1f%%" % (pt.soc * 100),
            lo=ELEC["soc_min"], hi=ELEC["soc_max"])
        bar(self.screen, self.font, x, y + 34, w, 14, pt.cap_kwh / pt.cap_max,
            C_GOOD, "SUPERCAP", "%4.0f Wh" % (pt.cap_kwh * 1000))
        bar(self.screen, self.font, x, y + 68, w, 14,
            min(1.0, pt.fly_kwh / max(0.01, pt.fly_kwh_max)),
            C_FLYWHEEL_BAR, "FLYWHEEL", "%5.0f RPM" % pt.fly_rpm)
        eng_pct = (pt.engine_seconds / pt.total_seconds * 100
                   if pt.total_seconds > 1 else 0.0)
        bar(self.screen, self.font, x, y + 102, w, 14, min(1.0, eng_pct / 5.0),
            C_RING_HOT if pt.engine_on else C_TEXT_DIM,
            "ENGINE RUNTIME", "%4.2f%%" % eng_pct)

        # --- EFFICIENCY STACK: the new fuel-free MPG systems, live ------------
        ex, ey, ew = 12, 210, 300
        panel(self.screen, ex, ey, ew, 372, alpha=205)
        self.screen.blit(self.fsmall.render("EFFICIENCY STACK  (fuel-free -> MPG)",
                         True, C_ACCENT), (ex + 8, ey + 6))
        fg_pct = math.tan(self.future_grade) * 100.0
        ahead = ("CLIMB ahead" if fg_pct > 1.2 else
                 ("DESCENT ahead" if fg_pct < -1.2 else "flat ahead"))
        eng_off = max(0.0, 100.0 - eng_pct)
        morphed = self.cd_eff < VEH["Cd"] - 1e-4
        hydro_pct = (1.0 - hydro_cd_factor()) * 100.0
        duct_pct = (1.0 - duct_cd_factor(self.cur_mph)) * 100.0
        rows = [
            ("ACTIVE AERO", "%.3f>%.3f%s" % (VEH["Cd"], self.cd_eff,
             " MORPH" if morphed else ""), C_GOOD if morphed else C_TEXT_DIM),
            ("FLOW DUCT", "-%.1f%% wake" % duct_pct, C_GOOD if duct_pct > 0.05 else C_TEXT_DIM),
            ("HYDRO SKIN", "-%.1f%% drag (RH%2.0f)" % (hydro_pct, HUMIDITY * 100),
             C_GOOD if hydro_pct > 0.05 else C_TEXT_DIM),
            ("MAG BEARINGS", "-%.0f%% roll (passive)" % (WHEEL_BEARING["roll_cut_frac"] * 100),
             C_GOOD),
            ("DRAG CAPTURE", "+%3.0f W" % (self.drag_kw * 1000),
             C_GOOD if self.drag_kw > 1e-4 else C_TEXT_DIM),
            ("WIND (parked)", ("+%3.0f W" % (self.wind_kw * 1000)) if self.wind_kw > 1e-4
             else "retracted", C_GOOD if self.wind_kw > 1e-4 else C_TEXT_DIM),
            ("PENDULUM RIM", ("+%3.0f W" % (self.pend_kw * 1000)) if self.pend_deploy > 0.05
             else "RETRACTED", C_GOOD if self.pend_kw > 1e-4 else C_TEXT_DIM),
            ("SOLAR ROOF", "+%3.0f W" % (self.solar_roof_kw * 1000),
             C_GOOD if self.solar_roof_kw > 1e-4 else C_TEXT_DIM),
            ("SUN LENS>BOIL", "+%.2f kW" % self.solar_boiler_kw,
             C_WARN if self.solar_boiler_kw > 1e-4 else C_TEXT_DIM),
            ("SUSPENSION", "+%3.0f W" % (self.susp_kw * 1000),
             C_GOOD if self.susp_kw > 1e-4 else C_TEXT_DIM),
            ("TEG (heat)", "+%3.0f W" % (self.teg_kw * 1000),
             C_GOOD if self.teg_kw > 1e-4 else C_TEXT_DIM),
            ("STEER REGEN", "+%3.0f W" % (self.steer_kw * 1000),
             C_GOOD if self.steer_kw > 1e-4 else C_TEXT_DIM),
            ("PCM HEAT BANK", "%3.0f C" % self.pcm_c,
             C_WARN if self.pcm_c > THERM["ambient_c"] + 3 else C_TEXT_DIM),
            ("PREDICT SOC", "%2.0f%% %s" % (self.soc_target * 100, ahead), C_ACCENT),
            ("ENGINE-OFF", "%4.1f%%" % eng_off,
             C_GOOD if eng_off > 95 else C_WARN),
            ("EXTRA HARVEST", "%4.0f Wh" % self.extra_wh, C_ACCENT),
            ("HV BUS", "%.0fV %4.0fA" % (self.elec["volts"], self.elec["amps"]),
             C_BAD if self.elec["over_limit"] else C_ACCENT),
            ("BUS LOSS", "%3.0f W I2R+inv" % (self.elec["loss_kw"] * 1000),
             C_WARN if self.elec["loss_kw"] > 0.15 else C_TEXT_DIM),
            ("ECU LOOP", "%.0f Hz  dt %.0fms" % (ECU["loop_hz"], 1000.0 / ECU["loop_hz"]),
             C_TEXT_DIM),
        ]
        for i, (lab, val, vc) in enumerate(rows):
            ry = ey + 24 + i * 18
            self.screen.blit(self.fsmall.render(lab, True, C_TEXT_DIM), (ex + 10, ry))
            vimg = self.fsmall.render(val, True, vc)
            self.screen.blit(vimg, (ex + ew - 12 - vimg.get_width(), ry))

        panel(self.screen, self.W // 2 - 410, self.H - 56, 820, 46, alpha=175)
        eng = "ENGINE FIRING" if pt.engine_on else "ENGINE OFF (electric)"
        ecol = C_RING_HOT if pt.engine_on else C_GOOD
        grade_pct = math.tan(self.cur_grade) * 100
        regen = " REGEN>>" if pt.regen_active else ""
        orc = self.therm["orc_kw"]
        tcol = C_BAD if self.phys.get("shutdown") else (
            C_WARN if self.phys.get("overheat") else C_TEXT)
        base = "%s%s   grade %+4.1f%%   %6.2f mi   " % (eng, regen, grade_pct, pt.miles)
        img = self.font.render(base, True, ecol)
        self.screen.blit(img, (self.W // 2 - 400, self.H - 52))
        self.screen.blit(self.font.render(
            "%3.0fC(boil %3.0f/pcm %3.0f) %s  %s %.1fbar  RECOV %.0fkW" % (
                self.temp_c, self.boiler_c, self.pcm_c, self.phys.get("status", "NORMAL"),
                self.fluid, self.therm.get("steam_bar", 0.0), orc), True, tcol),
            (self.W // 2 - 400 + img.get_width(), self.H - 52))
        # second row: ambient harvest + pedal-assist + working-fluid life
        aimg = self.font.render("HARVEST %3.0fW (%4.0f Wh)   " % (
            self.ambient_kw * 1000.0, self.harvest_wh), True, C_ACCENT)
        self.screen.blit(aimg, (self.W // 2 - 400, self.H - 32))
        pw = DIMS["seats"] * PEDAL_WATTS_PER_SEAT * self.pedal_engage
        pcol = C_GOOD if self.pedal_on else C_TEXT_DIM
        pimg = self.font.render("PEDAL %s %3.0fW   " % (
            "ON" if self.pedal_on else "off(K)", pw), True, pcol)
        self.screen.blit(pimg, (self.W // 2 - 400 + aimg.get_width(), self.H - 32))
        svc_due = self.fluid_life <= 0.12
        fcol = C_BAD if svc_due else C_GOOD
        ftxt = "%s life %3.0f%%%s" % (
            self.fluid, self.fluid_life * 100, " SERVICE(F)!" if svc_due else "")
        self.screen.blit(self.font.render(ftxt, True, fcol),
                         (self.W // 2 - 400 + aimg.get_width() + pimg.get_width(), self.H - 32))

        self.draw_pedal(self.W - 60, self.H - 150, "THR", self.throttle, C_GOOD)
        self.draw_pedal(self.W - 30, self.H - 150, "BRK", self.brake, C_BAD)

    def draw_pedal(self, x, y, label, val, color):
        h = 110
        pygame.draw.rect(self.screen, C_PANEL_HI, (x, y, 18, h), border_radius=4)
        fh = int(h * max(0, min(1, val)))
        pygame.draw.rect(self.screen, color, (x, y + h - fh, 18, fh), border_radius=4)
        self.screen.blit(self.fsmall.render(label, True, C_TEXT_DIM), (x - 4, y + h + 2))

    # ---- help ----
    def draw_help(self):
        if self.mode == self.MODE_CAR:
            lines = ["WHOLE-CAR VIEW  --  drag orbit | wheel zoom | right/middle pan | Shift fine",
                     "1 FULL  2 EXPLODED  3 ASSEMBLY  |  4/X SECTION-CUT | L (or LABELS chip) = names | R reset",
                     "hover any part = inspect its real spec | click = pin (or place next in assembly)",
                     "[TAB] cycles ENGINE PREVIEW > CAR VIEW > DRIVE"]
            y0, wide = 70, 830
        elif self.mode == self.MODE_PREVIEW:
            lines = ["ENGINE PREVIEW  --  drag orbit | wheel zoom-at-cursor | right/middle pan | Shift fine",
                     "1 FULL  2 EXPLODED  3 ASSEMBLY  |  4/X SECTION-CUT | L (or LABELS chip) = names | R reset",
                     "hover a part = inspect | click = pin (or place next in assembly)",
                     "P pause | -/= speed | [ ] pistons 1-10 | , . combustions 1-8 | F fluid | TAB car>drive"]
            y0, wide = 70, 830
        else:
            lines = ["DRIVE  --  W/Up: throttle  S/Down: brake  A/D: steer",
                     "R: Drive<->Reverse (stopped) | C: cruise | E: engine burst | G: downhill",
                     "K: passenger PEDAL-ASSIST | F: boiler fluid | SPACE: brake | TAB: preview | I: spec"]
            y0, wide = 204, 620   # under the speedo, clear of the bottom energy bars
        panel(self.screen, 12, y0 - 6, wide, len(lines) * 18 + 12)
        for i, ln in enumerate(lines):
            self.screen.blit(self.fsmall.render(ln, True, C_TEXT_DIM), (22, y0 + i * 18))

    # ---- info ----
    def draw_info(self):
        w = min(900, self.W - 80)
        x = (self.W - w) // 2
        s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        s.fill((4, 6, 10, 249))
        self.screen.blit(s, (0, 0))
        lines = []
        for title, body in INFO_SECTIONS:
            lines.append(("title", title))
            for b in body:
                lines.append(("body", b))
            lines.append(("gap", ""))
        total_h = len(lines) * 20 + 40
        self.info_scroll = max(0, min(self.info_scroll, max(0, total_h - (self.H - 120))))
        y = 60 - self.info_scroll
        head = self.fbig.render("FULL INFORMATIONAL SPECIFICATION", True, C_ACCENT)
        self.screen.blit(head, (x, 24))
        self.screen.blit(self.fsmall.render("scroll: mouse wheel    close: I",
                                            True, C_TEXT_DIM),
                         (x + head.get_width() + 16, 36))
        self.screen.set_clip(pygame.Rect(0, 56, self.W, self.H - 70))
        for kind, text in lines:
            if kind == "title":
                if 40 < y < self.H:
                    self.screen.blit(self.font.render(text, True, C_WARN), (x, y))
                y += 24
            elif kind == "body":
                if 40 < y < self.H:
                    self.screen.blit(self.fsmall.render(text, True, C_TEXT), (x + 14, y))
                y += 19
            else:
                y += 8
        self.screen.set_clip(None)

    # ---- real-life MPG chart ----
    def draw_mpg_chart(self):
        """Full-screen MPG-vs-speed chart from real ROAD-LOAD physics (not the
        arcade drive economy): shows 1-4 passengers, pedals on/off, the infinite-
        MPG band and why 80 mph is so much worse than a slow cruise."""
        W, H = self.W, self.H
        s = pygame.Surface((W, H), pygame.SRCALPHA)
        s.fill((4, 6, 10, 250))
        self.screen.blit(s, (0, 0))
        self.screen.blit(self.fbig.render("REAL-LIFE MPG  vs  SPEED", True, C_ACCENT), (30, 42))
        self.screen.blit(self.fsmall.render(
            "flat cruise, no brake regen | harvest ~%.0f W + drag-capture | %.0f kg | "
            "Cd %.3f>%.3f | A %.2f m2 | Crr %.4f>%.4f | fuel->wheel %.1f%% | close: M" % (
                ambient_harvest_w(50), VEH["curb_mass_kg"], VEH["Cd"], effective_cd(80),
                VEH["frontal_area_m2"], VEH["Crr"], effective_crr(), FUEL_TO_WHEEL_EFF * 100),
            True, C_TEXT_DIM), (34, 78))

        # ---- plot area ----
        plx, ply = 84, 150
        plw = int(W * 0.55) - plx
        plh = H - 160 - ply
        px0, py0 = plx, ply + plh
        VMAX, MMAX, inf_h = 85.0, MPG_PLOT_CAP, 44
        cols = {1: C_ACCENT, 2: C_GOOD, 3: C_WARN, 4: (255, 150, 90)}

        def X(v):
            return int(px0 + (v / VMAX) * plw)

        def Y(mpg):
            if mpg > MMAX:
                return int(ply - inf_h * 0.5)
            return int(py0 - (mpg / MMAX) * plh)

        pygame.draw.rect(self.screen, (10, 14, 20), (plx, ply - inf_h, plw, plh + inf_h))
        ib = pygame.Surface((plw, inf_h), pygame.SRCALPHA)
        ib.fill((40, 95, 62, 130))
        self.screen.blit(ib, (plx, ply - inf_h))
        self.screen.blit(self.fsmall.render(
            "SOLAR-SUSTAINED (>%d MPG)  --  external solar covers the load, NO fuel burned"
            % int(MMAX), True, C_GOOD), (plx + 8, ply - inf_h + 6))
        for mpg in (500, 1000, 1500, 2000):
            yy = Y(mpg)
            pygame.draw.line(self.screen, (28, 36, 48), (plx, yy), (plx + plw, yy), 1)
            self.screen.blit(self.fsmall.render("%4d" % mpg, True, C_TEXT_DIM), (plx - 44, yy - 7))
        self.screen.blit(self.fsmall.render("MPG", True, C_TEXT_DIM), (plx - 46, py0 - 16))
        for v in MPG_SPEEDS_MPH:
            xx = X(v)
            hot = (v == 80)
            pygame.draw.line(self.screen, (C_BAD if hot else (28, 36, 48)),
                             (xx, ply - inf_h), (xx, py0), 1)
            self.screen.blit(self.fsmall.render("%d" % v, True, C_BAD if hot else C_TEXT_DIM),
                             (xx - 6, py0 + 6))
        self.screen.blit(self.fsmall.render("speed (mph) ->", True, C_TEXT_DIM),
                         (px0 + plw - 96, py0 + 6))
        self.screen.blit(self.fsmall.render("80 mph: aero-dominated, worst MPG",
                         True, C_BAD), (X(80) - 236, py0 - 16))

        fine = [3, 5, 7, 10, 14, 20, 25, 32, 40, 50, 62, 75, 80, 85]
        # pedals-OFF: faint thin lines (n=1 and n=4 bound the passenger range)
        for n in (1, 4):
            c = tuple(int(x * 0.45) for x in cols[n])
            pts = [(X(v), Y(estimate_mpg(v, n, False))) for v in fine]
            pygame.draw.lines(self.screen, c, False, pts, 1)
        # pedals-ON: bold coloured line per passenger count
        for n in (1, 2, 3, 4):
            pts = [(X(v), Y(estimate_mpg(v, n, True))) for v in fine]
            pygame.draw.lines(self.screen, cols[n], False, pts, 2)
            for v in MPG_SPEEDS_MPH:
                pygame.draw.circle(self.screen, cols[n],
                                   (X(v), Y(estimate_mpg(v, n, True))), 3)
        # legend
        lx, lyy = plx + 12, ply + 8
        for n in (1, 2, 3, 4):
            pygame.draw.rect(self.screen, cols[n], (lx, lyy + (n - 1) * 18, 12, 10))
            self.screen.blit(self.fsmall.render("%d passenger%s pedaling" % (
                n, "" if n == 1 else "s"), True, C_TEXT), (lx + 18, lyy + (n - 1) * 18 - 2))
        self.screen.blit(self.fsmall.render("faint line = same, NO pedals",
                         True, C_TEXT_DIM), (lx, lyy + 4 * 18 + 2))

        # ---- data table (right) ----
        tx, ty = int(W * 0.57), 108
        def cell(mpg):
            return ">%4d" % int(MMAX) if mpg > MMAX else "%5.0f" % mpg
        self.screen.blit(self.font.render("REAL-LIFE ESTIMATE (MPG)", True, C_ACCENT), (tx, ty))
        hdr = "%-6s %4s %8s | %5s | %5s %5s %5s %5s" % (
            "speed", "aero", "gain", "noped", "p1", "p2", "p3", "p4")
        self.screen.blit(self.fsmall.render(hdr, True, C_TEXT_DIM), (tx, ty + 26))
        oy = ty + 44
        for v in MPG_SPEEDS_MPH:
            tot, aero, _ = road_load_watts(v, 1)
            row = "%-4dmph %3.0f%% %8s | %s | %s %s %s %s" % (
                v, aero / max(1e-6, tot) * 100, mpg_gain_label(v),
                cell(estimate_mpg(v, 1, False)),
                cell(estimate_mpg(v, 1, True)), cell(estimate_mpg(v, 2, True)),
                cell(estimate_mpg(v, 3, True)), cell(estimate_mpg(v, 4, True)))
            col = C_BAD if v == 80 else (C_GOOD if v <= 10 else C_TEXT)
            self.screen.blit(self.fsmall.render(row, True, col), (tx, oy))
            oy += 18

        aero80 = road_load_watts(80, 1)[1] / max(1e-6, road_load_watts(80, 1)[0]) * 100
        why = [
            "",
            "WHY  --  road power = (aero + rolling) x speed:",
            " - aero drag POWER grows as speed^3; rolling only as speed^1.",
            " - so ~2x the speed is ~8x the aero power -> MPG collapses.",
            "   SLOWER IS DRAMATICALLY BETTER (%.0f MPG @5 vs %.0f @80)." % (
                estimate_mpg(5, 1, False), estimate_mpg(80, 1, False)),
            "",
            "AI-AERO BODY: Cd %.2f, %.2f m2 frontal, %.0f kg, Crr %.4f --" % (
                VEH["Cd"], VEH["frontal_area_m2"], VEH["curb_mass_kg"], VEH["Crr"]),
            " expanded aero/rolling optimization cuts the road load again:",
            " 50 mph: %s -> %s MPG (%s); 80 mph: %s -> %s MPG (%s)." % (
                _fmt_mpg(estimate_mpg_baseline(50, 1, False)),
                _fmt_mpg(estimate_mpg(50, 1, False)), mpg_gain_label(50),
                _fmt_mpg(estimate_mpg_baseline(80, 1, False)),
                _fmt_mpg(estimate_mpg(80, 1, False)), mpg_gain_label(80)),
            "",
            "SOLAR (~%.0f W in sun): PV roof + film -- EXTERNAL energy, so below" % (
                (AMBIENT_HARVEST_W["solar"] + AMBIENT_HARVEST_W["solar_roof"]) * SUN),
            " ~10-25 mph it alone can cover the road load -> SOLAR-SUSTAINED, no",
            " fuel (the tyre/tribo harvest only RECLAIMS rolling loss, never free).",
            "",
            "PEDALS (%.0f W/seat) stack on top of that:" % PEDAL_WATTS_PER_SEAT,
            " - more passengers = more free watts = fuel-free to a higher speed.",
            " - at high speed pedals are a rounding error, so they RETRACT into",
            "   the frame (useless against ~%.0f%% aero load at 80 mph)." % aero80,
            "",
            "80 mph is 'bad' only vs slower cruise -- ~%.0f MPG is still huge" % estimate_mpg(80, 1, False),
            "for a car, but a gentle cruise makes 5-15x that. Ease off = win.",
        ]
        oy += 8
        for ln in why:
            c = C_ACCENT if ln.endswith(":") else (
                C_GOOD if "SOLAR-SUSTAINED" in ln or "SLOWER" in ln else C_TEXT_DIM)
            self.screen.blit(self.fsmall.render(ln, True, c), (tx, oy))
            oy += 17

    # ---------------------------------------------------------------- loop
    def run(self):
        print_startup_banner()
        while self.running:
            dt = min(0.05, self.clock.tick(60) / 1000.0)
            self.handle_events(dt)
            self.update(dt)
            self.draw()
        pygame.quit()


def print_startup_banner():
    print("=" * 70)
    print(" GmansRun V1.17 -- HOHEV-Rotary Gen 4 Standalone Digital Twin")
    print("=" * 70)
    print(" PREVIEW mode loads first: orbit the full 3D engine and watch every")
    print(" part work. TAB cycles ENGINE PREVIEW > CAR VIEW (the whole vehicle to")
    print(" real-life scale, its own full/exploded/assembly) > DRIVE. I = full spec.")
    print("-" * 70)
    print(" NEW EFFICIENCY STACK (all fuel-free, all automatic, all raise MPG):")
    print("   active morphing aero  |  dynamic regen suspension  |  thermoelectric TEG")
    print("   PCM heat bank (engine-off steam)  |  predictive control + dynamic SOC")
    print("   regen steering + seat/body piezo   -- see the DRIVE panel + info 6f")
    print("   SOLAR ROOF panel (+electric) + WINDSHIELD sun-concentrator lens (+boiler")
    print("   heat -> steam) -- sunshine into power, even engine-off  -- see info 6g")
    print("   SUPER-HYDROPHOBIC slip skin: rain beads/rolls off, plastron air-slip cuts")
    print("   drag (more in humid air) -- pure Cd reduction -> more MPG  -- see info 8b")
    print("   PASSIVE permanent-magnet (hard-magnetic, NOT electromagnetic) wheel bearings:")
    print("   near-zero friction, no coil power, stabilized -> lower Crr, more MPG (8c)")
    print("   PASSIVE DRAG-AIR CAPTURE: micro-grooves funnel leftover drag air to bladeless")
    print("   turbines (recover wasted drag) + PARKED WINDMILL tail-fin wind charging (6h)")
    print("   INERTIAL PENDULUM RIM: hanging bumper-skirt weights swing on accel/turn/incline")
    print("   changes and drive tiny generators; deploys at low speed, RETRACTS at high (6i)")
    print("   EM RAIL-RING DIRECT DRIVE: the wheel RIM is the rotor, a stator rail ring turns")
    print("   it gearlessly (no drive clutch) at ~97%% -> less energy/mile, more MPG (11)")
    print("   CAR VIEW shows all 4 rail-ring wheels with in-rim disc, rail, poles and labels")
    print("   SIZE-OPTIMIZED: optimize_harvest.py ran 300k+ tests to scale every harvester to")
    print("   its net-best size (benefit vs mass/drag) for max synergy -- +155.4%% cycle MPG (12)")
    print("   KINETIC->ELECTRICAL CHAIN re-optimized: generator/mechanical/gear to their")
    print("   ceilings -> ~46.5%% fuel->wheel, more charge per gallon (see info 13, 0-loss gap)")
    print("   STORAGE round-trips now modelled (correctness): fast->supercap, bulk->battery,")
    print("   overflow->flywheel -- optimal routing keeps effective round-trip ~96%% (14)")
    print("   ONE-PASS NETTING: harvest used the same instant for traction bypasses storage")
    print("   entirely; operating-point + control confirmed at optimum -- see info 14, 15")
    print("   THROUGH-BODY FLOW DUCT: straight front-grill->rear-diffuser pipe fills the rear")
    print("   wake to kill pressure drag (dominant at speed) -> big highway MPG gain (8d)")
    print("-" * 70)
    print(" Controls:")
    print("   TAB  cycle ENGINE PREVIEW > CAR VIEW > DRIVE   M  MPG chart   I  info   H  help")
    print("   PREVIEW VIEWS:  1=FULL  2=EXPLODED  3=ASSEMBLY  |  4 or X = SECTION-CUT toggle")
    print("   PREVIEW: drag=orbit  wheel=zoom-at-cursor  right/middle=pan")
    print("            Shift=fine camera control  L=labels  R=reset")
    print("            hover a part = read spec | click = pin / place next part")
    print("   PLAYBACK: P=pause  -/= view speed  [ ]=pistons 1-10  , . =combustions 1-8  F=fluid")
    print("   DRIVE:   W/Up=throttle  S/Down=brake  A/D=steer  C=cruise  E=engine burst")
    print("            G=downhill  SPACE=hard brake  K=passenger pedal-assist  F=boiler fluid")
    print("   ESC  quit")
    print("=" * 70)


def main():
    App().run()


if __name__ == "__main__":
    main()
