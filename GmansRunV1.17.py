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
    "wheel_gen_d_mm":       300.0,   # in-wheel axial-flux motor/generator
    "wheel_width_mm":       150.0,

    # --- Passenger pop-out pedal-assist trickle generators ---------------
    "seats":                  4,     # one pop-out pedal generator per seat
    "pedal_crank_d_mm":      92.0,   # pedal crank / flywheel
    "pedal_gen_d_mm":        70.0,   # mini trickle generator
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
FUEL_TO_WHEEL_EFF  = 0.44    # redesigned engine (brake ~0.40) + ORC recovery + low-
#                              loss SiC inverter/gen/drivetrain -> ~44% fuel-to-wheel
OCCUPANT_KG        = 80.0    # mass added per occupant beyond the driver
MPG_SPEEDS_MPH     = [5, 10, 25, 50, 80]
MPG_PLOT_CAP       = 2000.0  # chart clamps above this into the 'infinite' band
PEDAL_HELP_FRAC    = 0.05    # below this share of road load, pedals retract

# Firing / RPM behaviour
FIRING_IDLE_RPM   = 320.0     # always-spinning warm idle when vehicle moving
FIRING_GEN_RPM    = 3000.0    # generating RPM (chambers fire 1x / rev)
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
    "recovery_eff_cap": 0.55,  # ceiling on effective heat->electric conversion
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
    "indicated_eff":               0.44,  # higher-compression, cleaner burn
    "mechanical_eff":              0.93,  # low-friction redesign (little to no loss)
    "generator_eff":               0.95,  # better axial-flux PM generator
    "gear_mesh_eff":               0.975, # ground/lapped teeth
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
# drop Cd to ~0.09 and shrink the frontal area of a narrow tandem cabin. Combined
# with a generative-lattice ultralight structure and low-rolling-resistance tyres,
# this slashes the road load -- the single biggest lever on real-world MPG.
VEH = {
    "curb_mass_kg":    765.0,   # <770 kg: carbon monocoque + STRUCTURAL battery + driver
    "Cd":              0.09,    # AI shape-optimized teardrop + boat-tail, covered wheels
    "frontal_area_m2": 1.68,    # narrow, low tandem cabin (active air dam + underbody)
    "Crr":             0.0032,  # airless metamaterial tyres, active pressure, sealed floor
    "air_density":     1.225,
    "g":               9.81,
    "wheel_radius_m":  0.32,
    "drivetrain_eff":  0.95,    # motor + inverter chain (low-loss SiC inverters)
    "max_motor_kw":    240.0,   # 4 in-wheel axial-flux motors combined
    "max_brake_n":     9000.0,
    "regen_frac":      0.87,    # braking recovery with axial-flux + supercap buffer
}

# MULTI-LAYER AMBIENT HARVEST -- the compounding, fuel-free trickle that runs even
# with the engine off and the wheels coasting. No single magic part; every joule is
# attacked from several angles at once. Favorable-daylight estimates (watts):
AMBIENT_HARVEST_W = {
    "solar":        260.0,   # ~1.5 m2 quantum-dot film over the whole upper surface
    "suspension":    80.0,   # linear electromagnetic regenerative dampers (real roads)
    "triboelectric": 40.0,   # TENG films on underbody + wheel wells
    "tire":          22.0,   # airless metamaterial tyres with embedded harvesters
}

# Electrical storage. HEADROOM IS KING: hold a narrow 15-55% SOC window so there
# is always empty room to swallow braking + downhill + ambient harvest -- never
# chase a full battery. A big supercapacitor bank takes every peak charge/drain.
ELEC = {
    "batt_kwh":        40.0,    # structural battery integrated into the monocoque
    "soc_min":         0.15,    # never held high; max regen headroom
    "soc_max":         0.55,
    "soc_start":       0.35,    # start mid-window
    "supercap_kwh":    0.65,    # 400+ Farad bank, real-time charge/drain
}

# Engine control thresholds (keeps engine runtime < ~2% of drive time).
# The rotary is a RARE high-efficiency burst generator only: it fires when the
# battery + flywheel + live harvesting cannot meet demand, then stops the instant
# EITHER the battery target OR the heat target is reached -- once the block is hot
# the trickle steam-regen keeps harvesting with the engine OFF.
ENGINE_ON_SOC    = 0.20       # fire a burst when buffers run low
ENGINE_OFF_SOC   = 0.50       # stop once the battery is topped to here, OR...
ENGINE_OFF_TEMP_C = 118.0     # ...once the block is hot enough to keep making steam
ENGINE_CRIT_SOC  = 0.13       # safety floor: refire even when hot if this low


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
    """A road WHEEL with an in-hub axial-flux GENERATOR. On braking / downhill the
    in-wheel motors run as generators and pour momentum back into the caps +
    battery. Shown to the side as the regenerative harvest output."""
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
    # in-hub axial-flux GENERATOR (charges on brake / downhill)
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
    parts.append(Part("gen", "Axial-Flux Generator (charges battery/caps)", gm,
        ["Function: turns transmission output into electricity -- this output",
         "CHARGES the batteries + capacitors; it does NOT drive the wheels",
         "(the wheels run on in-wheel motors from the battery).",
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
             "Idle/cruise = 1 combustion/rev; full power = 8 combustions/rev",
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
    parts.append(Part("wheelgen", "Regen Wheel-Generator (brake/downhill)",
        _regen_wheel(0.0, r_out),
        ["Function: the road wheels carry in-hub axial-flux motor/GENERATORS.",
         "On braking and downhill they ENGAGE as generators and pour recovered",
         "momentum straight into the supercaps + battery (regenerative braking).",
         "This is the primary momentum harvest; the steam loop is the primary",
         "heat harvest. Together they maximize charge with minimal fuel."],
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
         "Total ~%.0f W charges even coasting with the engine OFF -- below ~10-25 mph" % ambient_harvest_w(0),
         "it alone can cover the road load (infinite MPG). This is a core MPG lever."],
        order=nextord(), explode=(0.0, 1.4, 0.0), color=(70, 120, 190)))

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
         "Inertia ~%.2f kg.m^2 ; ~1.9 kWh @ %.0f rpm; magnetic bearings" % (ifw, FLYWHEEL_MAX_RPM)],
        order=nextord(), explode=(0, 0, 0.42 + n_pistons * 0.42 + 0.12), color=C_FLYWHEEL))

    return parts


class EngineRenderer:
    """Projects + paints the spec'd engine Parts. Supports full / exploded /
    assembly views with an optional cross-SECTION CUT toggle, mouse hover-
    picking, highlight + hover-pop, and the assembly 'puzzle'."""

    def __init__(self):
        self.n_pistons = 1
        self.parts = build_engine_parts(self.n_pistons)
        self.az = 0.65
        self.el = 0.50
        self.dist = 1.55
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
        n = max(1, min(10, n))
        if n == self.n_pistons:
            return
        self.n_pistons = n
        self.parts = build_engine_parts(n)
        self.pop = np.zeros(len(self.parts))
        self.hovered = None
        self.selected = None
        self.assembled = len(self.parts)

    def reset_view(self):
        self.az, self.el, self.dist = 0.65, 0.50, 1.55
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
               selector_radius=None):
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
                _label(surf, label_font, text, (lx, ly2), accent=(tag == "active"))

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
            key.startswith("ring") or key.startswith("clutch")
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


def road_load_watts(v_mph, n_pass=1):
    """Steady-cruise road power at the wheels (real road-load physics).
    Returns (total_W, aero_W, roll_W). aero grows as v^3, rolling as v^1."""
    v = max(0.0, v_mph) * 0.44704
    m = VEH["curb_mass_kg"] + max(0, n_pass - 1) * OCCUPANT_KG
    f_aero = 0.5 * VEH["air_density"] * VEH["Cd"] * VEH["frontal_area_m2"] * v * v
    f_roll = VEH["Crr"] * m * VEH["g"]
    return (f_aero + f_roll) * v, f_aero * v, f_roll * v


def ambient_harvest_w(v_mph=0.0):
    """Total fuel-free AMBIENT harvest (watts): solar film + electromagnetic
    suspension dampers + triboelectric TENG + tyre harvesters. Solar is constant;
    the motion-based sources scale up a little with speed (rougher/faster = more)."""
    h = AMBIENT_HARVEST_W
    motion = 0.6 + 0.4 * clamp(v_mph / 55.0)      # 0.6 parked .. 1.0 at highway
    return (h["solar"]
            + (h["suspension"] + h["triboelectric"] + h["tire"]) * motion)


def estimate_mpg(v_mph, n_pass=1, pedals=False):
    """Real-life fuel-MPG estimate for a STEADY FLAT cruise (no braking/downhill
    regen assumed -- honest). Gasoline only makes electricity, so:
      fuel power = (road power - FREE harvest) / fuel-to-wheel efficiency,
    where free harvest = always-on ambient systems (solar/suspension/TENG/tyre)
    plus, if engaged, the passengers' pedals. If free power covers the whole road
    load, no fuel burns at all -> infinite MPG. Otherwise MPG = 1 / gal-per-mile."""
    if v_mph <= 0.0:
        return float("inf")
    p_wheel, _, _ = road_load_watts(v_mph, n_pass)
    free_w = ambient_harvest_w(v_mph)
    if pedals:
        free_w += n_pass * PEDAL_WATTS_PER_SEAT
    deficit = p_wheel - free_w
    if deficit <= 0.0:
        return float("inf")                        # harvest sustains it -> no fuel
    kwh_per_mile = (deficit / 1000.0) / v_mph      # kW * hours-per-mile
    gal_per_mile = kwh_per_mile / (FUEL_KWH_PER_GAL * FUEL_TO_WHEEL_EFF)
    return 1.0 / gal_per_mile


def pedals_help(v_mph, n_pass=1):
    """True while pedal power is a meaningful share of the road load; below that
    the pop-out pedals RETRACT into the frame (useless at high speed)."""
    p_wheel, _, _ = road_load_watts(v_mph, n_pass)
    if p_wheel <= 1e-6:
        return True
    return (n_pass * PEDAL_WATTS_PER_SEAT) >= PEDAL_HELP_FRAC * p_wheel


def _fmt_mpg(x):
    return " inf" if x == float("inf") else "%4.0f" % x


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
    # Vapour-pressure ramp from the BOILER FLUID temperature. A low-boil fluid
    # starts making pressure much lower, so it harvests low-grade heat water can't.
    steam_span = max(1.0, fl["full_c"] - fl["boil_c"])
    steam_ramp = clamp((boiler_c - fl["boil_c"]) / steam_span)
    steam_bar = fl["max_bar"] * steam_ramp
    # DUAL-CHAMBER, MULTI-STAGE (compound) reuse: the pressure is routed and
    # rerouted through several internal stages until weak, with two chambers
    # alternating so there is no dead stroke -> higher effective conversion.
    compound_gain = 1.0 + (THERM["compound_stages"] - 1) * THERM["stage_gain"]
    recovery_eff = min(THERM["recovery_eff_cap"],
                       fl["gen_eff"] * compound_gain * THERM["dual_chamber_gain"])
    steam_draw_kw = fl["capacity_kw"] * steam_ramp     # heat the boiler turns to vapour
    # aging fluid recovers a bit less until it is serviced (fluid_health 0..1)
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
        """Route signed kWh through supercap, battery, then flywheel."""
        self.cap_kwh += dE
        fly_dE = 0.0
        if self.cap_kwh > self.cap_max:
            spill = self.cap_kwh - self.cap_max
            self.cap_kwh = self.cap_max
            room = (ELEC["soc_max"] - self.soc) * self.batt_kwh
            into_batt = min(spill, max(0.0, room))
            self.soc += into_batt / self.batt_kwh
            fly_dE = spill - into_batt
        elif self.cap_kwh < 0.0:
            need = -self.cap_kwh
            self.cap_kwh = 0.0
            avail = max(0.0, self.soc * self.batt_kwh)
            from_batt = min(need, avail)
            self.soc -= from_batt / self.batt_kwh
            fly_dE = -(need - from_batt)

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

    def update(self, dt, v_mph, road_power_kw, moving, combustions=8):
        """road_power_kw: + = power demanded at the wheels, - = power available
        from braking / downhill (regen)."""
        self.total_seconds += dt
        self.miles += v_mph * (dt / 3600.0)

        # engine burst controller: fire only when buffers are low, and STOP as soon
        # as EITHER the battery target OR the heat target is reached -- once the
        # block is hot the trickle steam-regen keeps charging with the engine OFF.
        if self.force_timer > 0.0:
            self.force_timer = max(0.0, self.force_timer - dt)
            self.engine_on = True
        elif not self.engine_on and self.soc < ENGINE_ON_SOC \
                and (self.block_temp_c < ENGINE_OFF_TEMP_C or self.soc < ENGINE_CRIT_SOC):
            self.engine_on = True
        elif self.engine_on and (self.soc >= ENGINE_OFF_SOC
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

        net_kw = engine_kw + regen_kw - trac_kw
        dE = net_kw * (dt / 3600.0)                  # kWh this step

        # route energy: supercap first, then battery, then flywheel
        self.route_storage(dE, dt)

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

    def grade_at(self, s):
        g = (0.045 * math.sin(s / 220.0)
             + 0.030 * math.sin(s / 70.0 + 1.3)
             + 0.018 * math.sin(s / 33.0 + 2.1))
        return math.atan(g)

    def update(self, dt, throttle, brake, pt, force_grade=0.0, combustions=8):
        m = VEH["curb_mass_kg"]
        g = VEH["g"]
        grade = self.grade_at(self.dist_m) + force_grade
        sgn = 1.0 if self.v > 0.02 else (-1.0 if self.v < -0.02 else 0.0)
        self.regen_brake = False

        # resistive forces (signed along the forward direction)
        f_aero = -0.5 * VEH["air_density"] * VEH["Cd"] * VEH["frontal_area_m2"] * self.v * abs(self.v)
        f_roll = -sgn * VEH["Crr"] * m * g * math.cos(grade)
        f_grav = -m * g * math.sin(grade)

        # cruise control (forward only): add energy only when needed
        if self.cruise and self.gear > 0:
            err = self.cruise_set - self.v
            throttle = max(0.0, min(1.0, err * 0.6))
            if err < -0.5:
                brake = min(1.0, -err * 0.25)
                self.regen_brake = True

        fmag = throttle * VEH["max_motor_kw"] * 1000.0 / max(abs(self.v), 2.0)
        fmag = min(fmag, VEH["max_motor_kw"] * 1000.0 / 3.0)
        f_motor = fmag * self.gear                 # drive in the selected gear
        f_brake = -sgn * brake * VEH["max_brake_n"]

        self.v += (f_motor + f_brake + f_aero + f_roll + f_grav) / m * dt
        if self.gear < 0:
            self.v = max(self.v, self.REVERSE_MAX)
        # don't let braking drag the car backwards through a standstill
        if brake > 0 and throttle < 0.05 and abs(self.v) < 0.3:
            self.v = 0.0
        self.dist_m += self.v * dt

        demand_w = abs(f_motor) * abs(self.v) if throttle > 0.01 else 0.0
        regen_w = brake * VEH["max_brake_n"] * abs(self.v)
        if grade < 0:
            regen_w += m * g * (-math.sin(grade)) * abs(self.v) * 0.6
        road_power_kw = (demand_w - regen_w) / 1000.0

        v_mph = self.v * 2.23694
        pt.regen_active = regen_w > 100.0
        pt.update(dt, abs(v_mph), road_power_kw, moving=abs(self.v) > 0.3,
                  combustions=combustions)
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
        "harvest (solar/suspension/TENG/tyre), and even takes free passenger PEDAL",
        "power. Mixed real-world ~550-1,100 MPG; favorable 800-1,800+ (see 6c-6e, M).",
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
        "REGEN WHEELS: the in-wheel axial-flux motors run as GENERATORS on braking",
        "and downhill, pouring momentum straight into the supercaps + battery. This",
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
        "batteries + capacitors -- it does NOT drive the wheels. The wheels run on",
        "in-wheel motors fed from the battery (this engine is a range-extender).",
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
        "   friction, magnetic-bearing internals -> ~%.0f%% fuel-to-wheel, little loss." % (
            FUEL_TO_WHEEL_EFF * 100),
        "2) RIGHT-SIZED LEAN BURN: when the burner fires it runs at 1 combustion/rev",
        "   -- a tiny generator sipping fuel, sustained by the recovery, not oversized.",
        "3) HEAT REGEN: the ammonia %d-stage steam loop turns trapped waste heat" % THERM['compound_stages'],
        "   back into electricity, and keeps charging even after the burner stops.",
        "4) MOMENTUM REGEN: in-wheel generators recover ~%.0f%% of brake + downhill;" % (VEH['regen_frac'] * 100),
        "   descents also wind the free-floating tungsten flywheel via the clutch.",
        "5) KILLED THE DRAG: AI-optimized body Cd %.2f, %.0f kg, so cruising asks only" % (
            VEH['Cd'], VEH['curb_mass_kg']),
        "   a fraction of the power the other harvests cover outright.",
        "6) AMBIENT HARVEST: ~%.0f W of always-on solar + suspension + TENG + tyre." % ambient_harvest_w(0),
        "7) FREE HUMAN WATTS: passenger pedals trickle in more fuel-free charge.",
        "Real-life estimate (M chart): a gentle <=10 mph cruise = INFINITE MPG on",
        "harvest alone; 50 mph = ~%.0f MPG (up to ~%.0f with 4 pedalers, was 242);" % (
            estimate_mpg(50, 1, False), estimate_mpg(50, 4, True)),
        "80 mph = ~%.0f MPG. Mixed real-world ~550-1,100 MPG, favorable 800-1,800+ --" % estimate_mpg(80, 1, False),
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
        "the passengers' pedals (%.0f W each) cover it ALL -> INFINITE MPG, zero fuel." % PEDAL_WATTS_PER_SEAT,
        "More passengers = more free watts = infinite up to a higher speed (their",
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
    ("6e. MULTI-LAYER AMBIENT HARVEST (compound every joule)", [
        "No single magic part -- every joule is attacked from several angles at once,",
        "always on and FUEL-FREE, charging even while coasting with the engine off:",
        " - SOLAR: quantum-dot film over the whole upper surface (~%.0f W good sun)." % AMBIENT_HARVEST_W["solar"],
        " - SUSPENSION: linear electromagnetic regen dampers harvest road bumps",
        "   (~%.0f W on real roads) while actively controlling the ride." % AMBIENT_HARVEST_W["suspension"],
        " - TRIBOELECTRIC (TENG) underbody/wheel-well film (~%.0f W) + airless" % AMBIENT_HARVEST_W["triboelectric"],
        "   metamaterial TYRE harvesters (~%.0f W)." % AMBIENT_HARVEST_W["tire"],
        "Total ~%.0f W. It looks small, but it runs 24/7: below ~10-25 mph it ALONE" % ambient_harvest_w(0),
        "covers the whole road load -> INFINITE MPG with no pedals and no fuel.",
        "DOWNHILL FLYWHEEL: on descents, gravity + inertia wind the free-floating",
        "%.0f kg tungsten flywheel through the low-slip clutch (no combustion), and" % DIMS['flywheel_mass_kg'],
        "braking regen recovers ~%.0f%% via the axial-flux motors into the supercaps." % (VEH['regen_frac'] * 100),
        "HEADROOM IS KING: the battery is held in a tight %.0f-%.0f%% window so there" % (
            ELEC['soc_min'] * 100, ELEC['soc_max'] * 100),
        "is always empty room to swallow the next brake, hill and harvest. The rotary",
        "fires only when battery + flywheel + harvest cannot meet demand, then stops.",
        "Mixed real-world ~550-1,100 MPG; favorable slow/harvest-rich routes 800-1,800+.",
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
    ]),
    ("8. VEHICLE + AI-OPTIMIZED AERO BODY (kill the drag)", [
        "The single biggest real-world MPG lever is AERO DRAG, so the body was shape-",
        "optimized in an AI generative-design + CFD loop to the physical minimum: a",
        "low, narrow TEARDROP cabin with a long BOAT-TAIL, faired/covered wheels and",
        "a fully sealed flat underbody. That drops Cd to ~%.2f (from ~0.13) and the" % VEH['Cd'],
        "frontal area to %.2f m^2 -- roughly HALVING the aero force at any speed." % VEH['frontal_area_m2'],
        "A generative-lattice ultralight structure cuts curb mass to ~%.0f kg and" % VEH['curb_mass_kg'],
        "low-rolling-resistance tyres take Crr to %.4f, so the whole road load is" % VEH['Crr'],
        "tiny. Four in-wheel axial-flux motors (~%.0f kW) with low-loss SiC inverters" % VEH['max_motor_kw'],
        "and magnetic decoupling give true zero-drag coasting.",
        "Net: real-life road load at 50 mph fell by ~35%%, so MPG at 50 mph roughly",
        "DOUBLED (see the M chart) -- and slower still is far better again.",
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
    ("CONTROLS", [
        "TAB ... switch PREVIEW <-> DRIVE     M ... MPG chart   I ... info   H ... help",
        "PREVIEW VIEWS:  1 = FULL   2 = EXPLODED   3 = ASSEMBLY",
        "SECTION-CUT:  4 or X = toggle the cross-section over FULL / EXPLODED",
        "PREVIEW: drag orbit | wheel zoom-at-cursor | right/middle pan | L labels",
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
    MODE_PREVIEW = 0
    MODE_DRIVE = 1

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
        self.pt = Powertrain()
        self.world = DriveWorld()

        self.mode = self.MODE_PREVIEW
        self.engine_angle = 0.0
        self.paused = False
        self.speeds = [0.05, 0.15, 0.5, 1.0]
        self.speed_idx = 3
        self.combustions = 1                       # 1..8 combustions per rev
        self._eff_comb = 1                          # combustions actually fired now
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
        self.fluid = DEFAULT_FLUID                  # selected closed-loop fluid
        self.fluid_kwh = 0.0                         # recovered energy on this charge
        self.fluid_life = 1.0                        # 1..0 until a fluid change is due
        self.fluid_miles0 = 0.0                      # odometer at last fluid change
        self.pedal_on = False                        # passenger pedal-assist engaged
        self.pedal_engage = 0.0                      # ramp 0..1
        self.pedal_wh = 0.0                          # cumulative pedal energy (Wh)
        self.ambient_kw = 0.0                        # live multi-layer ambient harvest
        self.harvest_wh = 0.0                        # cumulative ambient harvest (Wh)
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
        self.show_labels = True
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
        self.bg = None
        self._rebuild_bg()

    def _rebuild_bg(self):
        self.bg = pygame.Surface((self.W, self.H))
        vgradient(self.bg, BG_TOP, BG_BOT)
        self._sky = pygame.Surface((self.W, self.H))   # cached drive sky gradient
        vgradient(self._sky, C_SKY1, C_SKY2)

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
                    self.mode = 1 - self.mode
                elif e.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif e.key == pygame.K_m:
                    self.show_mpg = not self.show_mpg      # real-life MPG chart
                elif e.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif e.key == pygame.K_l:
                    self.show_labels = not self.show_labels
                elif e.key == pygame.K_r and self.mode == self.MODE_PREVIEW:
                    self.renderer.reset_view()
                elif self.mode == self.MODE_PREVIEW and e.key in (
                        pygame.K_1, pygame.K_2, pygame.K_3):
                    self.renderer.set_view({pygame.K_1: "full", pygame.K_2: "exploded",
                                            pygame.K_3: "assembly"}[e.key])
                elif self.mode == self.MODE_PREVIEW and e.key in (pygame.K_4, pygame.K_x):
                    self.renderer.toggle_section()      # section CUT is a toggle
                elif self.mode == self.MODE_PREVIEW and self.renderer.view == "assembly" \
                        and e.key in (pygame.K_n, pygame.K_SPACE, pygame.K_RIGHT):
                    self.renderer.assembly_next()
                elif self.mode == self.MODE_PREVIEW and self.renderer.view == "assembly" \
                        and e.key in (pygame.K_b, pygame.K_LEFT, pygame.K_BACKSPACE):
                    self.renderer.assembly_prev()
                elif self.mode == self.MODE_PREVIEW and self.renderer.view == "assembly" \
                        and e.key == pygame.K_f:
                    self.renderer.assembly_all()
                elif self.mode == self.MODE_PREVIEW and self.renderer.view == "assembly" \
                        and e.key == pygame.K_0:
                    self.renderer.assembly_clear()
                elif e.key == pygame.K_p and self.mode == self.MODE_PREVIEW:
                    self.paused = not self.paused
                elif e.key == pygame.K_MINUS and self.mode == self.MODE_PREVIEW:
                    self.speed_idx = max(0, self.speed_idx - 1)
                    self.paused = False
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS) and self.mode == self.MODE_PREVIEW:
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
                elif e.button in (4, 5) and self.mode == self.MODE_PREVIEW:
                    rect = pygame.Rect(0, 34, self.W, self.H - 34)
                    factor = 0.82 if e.button == 4 else 1.22
                    self.renderer.zoom_at(factor, e.pos, rect)
            elif e.type == pygame.MOUSEBUTTONUP:
                if (e.button == 1 and self._drag_slider is None and self._press_pos is not None
                        and self.mode == self.MODE_PREVIEW and not self.show_info):
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
                elif self.mode == self.MODE_PREVIEW and not self.show_info:
                    fine = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    if self.dragging:
                        self.renderer.orbit(dx, dy, fine=fine)
                    elif self.panning:
                        self.renderer.pan_by(dx, dy, fine=fine)
            elif e.type == pygame.MOUSEWHEEL:
                if self.show_info:
                    self.info_scroll = max(0, self.info_scroll - e.y * 30)
                elif self.mode == self.MODE_PREVIEW:
                    rect = pygame.Rect(0, 34, self.W, self.H - 34)
                    factor = 0.82 ** e.y if e.y > 0 else 1.22 ** (-e.y)
                    self.renderer.zoom_at(factor, pygame.mouse.get_pos(), rect)

        if self.mode == self.MODE_DRIVE and not self.show_info:
            t_press = keys[pygame.K_w] or keys[pygame.K_UP]
            b_press = keys[pygame.K_s] or keys[pygame.K_DOWN] or keys[pygame.K_SPACE]
            self.throttle += ((1.0 if t_press else 0.0) - self.throttle) * min(1, dt * 6)
            self.brake += ((1.0 if b_press else 0.0) - self.brake) * min(1, dt * 8)
            self.world.steer = ((1 if keys[pygame.K_d] else 0)
                                - (1 if keys[pygame.K_a] else 0))

    def _preview_click(self):
        """Click in preview: pin/unpin a part (full/exploded) or place next
        part (assembly puzzle)."""
        if self.renderer.view == "assembly":
            self.renderer.assembly_next()
        else:
            h = self.renderer.hovered
            self.renderer.selected = None if (h is None or self.renderer.selected == h) else h

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
        rpm = {"piston": self.piston_rpm, "core": core, "fly": self.fly_rpm_v,
               "gen": gen, "out": out, "super": core * (2.2 + 2.0 * sc),
               "steam": steam_spin, "recov2": core * 1.8 + (300.0 if ep > 0.02 else 0.0),
               "wheel": wheel_rpm, "pedal": pedal_spin, "static": 0.0}
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
            slip_load = max(0.15 if abs(self.world.v) > 1 else 0.0, self.brake)
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
        boiler_leak = THERM["boiler_leak_kw_c"] * max(0.0, self.boiler_c - THERM["ambient_c"])
        self.boiler_c += (heat_boiler - steam_draw - boiler_leak) / THERM["boiler_mass_kj_c"] * dt
        self.boiler_c = max(THERM["ambient_c"], min(PHYS["max_temp_c"], self.boiler_c))

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
                      "boiler_c": self.boiler_c, "exh2_kw": self.phys["exh_stage2_kw"]}
        return self.phys["orc_kw"]

    # ---------------------------------------------------------------- update
    def update(self, dt):
        self.renderer.tick(dt)
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
        # MPG strategy: the range-extender runs LEAN at 1 combustion/rev -- a small,
        # right-sized generator sipping fuel is far more efficient (much higher MPG)
        # than firing oversized 8/rev bursts. Higher rates are for the preview bench.
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
            # Multi-layer AMBIENT harvest (solar + suspension + TENG + tyre): always
            # on, fuel-free, and it charges even coasting with the engine off.
            self.ambient_kw = ambient_harvest_w(self.cur_mph) / 1000.0
            self.harvest_wh += self.ambient_kw * 1000.0 * (dt / 3600.0)
            # Physics: main generator + recovered heat + pedal + ambient trickle.
            charge_kw = self.therm["main_kw"] + orc + pedal_kw + self.ambient_kw
            self.pt.route_storage(charge_kw * (dt / 3600.0), dt)
            self.pedal_wh += pedal_kw * 1000.0 * (dt / 3600.0)
            self.pt.flow["pedal"] = pedal_kw
            self.pt.flow["ambient"] = self.ambient_kw
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
        if self.mode == self.MODE_PREVIEW:
            vname = {"full": "FULL", "exploded": "EXPLODED", "assembly": "ASSEMBLY"}
            cut = " +SECTION-CUT" if self.renderer.section else ""
            mode = "PREVIEW [%s%s]" % (vname.get(self.renderer.view, "FULL"), cut)
        else:
            mode = "DRIVE  (simulation)"
        self.screen.blit(self.fbig.render("GmansRun V1.17", True, C_ACCENT), (12, 2))
        t = self.font.render("MODE: %s   [TAB] switch  [M] MPG chart  [I] info  [H] help" % mode,
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
        x, y = 12, 42
        tabs = (("1", "FULL", "full"), ("2", "EXPLODED", "exploded"),
                ("3", "ASSEMBLY", "assembly"), ("4", "SECTION-CUT", "section"))
        for key, name, mode in tabs:
            cur = self.renderer.section if mode == "section" else (self.renderer.view == mode)
            img = self.fsmall.render("%s %s" % (key, name), True,
                                     C_TEXT if cur else C_TEXT_DIM)
            w = img.get_width() + 18
            panel(self.screen, x, y, w, 22, alpha=225 if cur else 150)
            if cur:
                pygame.draw.rect(self.screen, C_ACCENT, (x, y, w, 22), 1, border_radius=6)
            self.screen.blit(img, (x + 9, y + 4))
            x += w + 6
        if self.renderer.view == "assembly":
            prog = "Assembled %d / %d   (click or N = place next, B = back, F = all)" % (
                self.renderer.assembled, len(self.renderer.parts))
            self.screen.blit(self.fsmall.render(prog, True, C_WARN), (x + 6, y + 4))

    def draw_spec_card(self):
        part = self.renderer.active_part()
        placing = None
        if part is None and self.renderer.view == "assembly":
            placing = self.renderer.placing_part()
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
        if self.renderer.selected is not None and placing is None:
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
        self.draw_car()
        self.draw_drive_hud()
        iw, ih = 360, 300
        ir = pygame.Rect(self.W - iw - 12, 44, iw, ih)
        panel(self.screen, ir.x, ir.y, ir.w, ir.h, alpha=180)
        self.screen.blit(self.fsmall.render("LIVE ENGINE", True, C_ACCENT),
                         (ir.x + 8, ir.y + 4))
        self.renderer.render(self.screen,
                             pygame.Rect(ir.x, ir.y + 16, ir.w, ir.h - 16),
                             self.group_angle, self.pt.firing_glow,
                             show_labels=False, view_override="full",
                             injection=self._injection, heat=self._heat_norm(),
                             selector_radius=output_gear_mesh_radius(self.kin["gear"] - 1))

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

    def draw_car(self):
        # AI-optimized aerodynamic body: a low, narrow TEARDROP cabin with a long
        # boat-tail and FAIRED (covered) wheels -- drawn instead of a boxy sedan to
        # show the shape that gets Cd ~0.09 and the low road load / high MPG.
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
        # aero badge
        self.screen.blit(self.fsmall.render("AI-AERO  Cd %.2f  A %.2fm2" % (
            VEH["Cd"], VEH["frontal_area_m2"]), True, c_glass), (cx - 62, cy - 60))
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
        self.screen.blit(self.font.render("%3.0fC(boil %3.0f) %s  %s %.1fbar  RECOV %.0fkW" % (
            self.temp_c, self.boiler_c, self.phys.get("status", "NORMAL"),
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
        if self.mode == self.MODE_PREVIEW:
            lines = ["drag orbit | wheel zoom-at-cursor | right/middle pan | Shift = fine control",
                     "1 FULL  2 EXPLODED  3 ASSEMBLY  |  4 / X = SECTION-CUT toggle | L labels | R reset",
                     "hover a part = inspect | click = pin (or place next in assembly)",
                     "P pause | -/= speed | [ ] pistons 1-10 | , . combustions 1-8 | F boiler fluid"]
            y0, wide = 70, 790
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
        self.screen.blit(self.fbig.render("REAL-LIFE MPG  vs  SPEED", True, C_ACCENT), (30, 18))
        self.screen.blit(self.fsmall.render(
            "steady flat cruise; ambient harvest (~%.0f W) ON, NO braking regen  --  %.0f kg, "
            "Cd %.2f, A %.1f m2, Crr %.4f, fuel->wheel %.0f%%   (close: M)" % (
                ambient_harvest_w(50), VEH["curb_mass_kg"], VEH["Cd"],
                VEH["frontal_area_m2"], VEH["Crr"], FUEL_TO_WHEEL_EFF * 100),
            True, C_TEXT_DIM), (34, 58))

        # ---- plot area ----
        plx, ply = 84, 130
        plw = int(W * 0.55) - plx
        plh = H - 160 - ply
        px0, py0 = plx, ply + plh
        VMAX, MMAX, inf_h = 85.0, MPG_PLOT_CAP, 44
        cols = {1: C_ACCENT, 2: C_GOOD, 3: C_WARN, 4: (255, 150, 90)}

        def X(v):
            return int(px0 + (v / VMAX) * plw)

        def Y(mpg):
            if mpg == float("inf") or mpg > MMAX:
                return int(ply - inf_h * 0.5)
            return int(py0 - (mpg / MMAX) * plh)

        pygame.draw.rect(self.screen, (10, 14, 20), (plx, ply - inf_h, plw, plh + inf_h))
        ib = pygame.Surface((plw, inf_h), pygame.SRCALPHA)
        ib.fill((40, 95, 62, 130))
        self.screen.blit(ib, (plx, ply - inf_h))
        self.screen.blit(self.fsmall.render(
            "INFINITE MPG  --  pedals/regen cover the whole load, NO fuel burned",
            True, C_GOOD), (plx + 8, ply - inf_h + 6))
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
            return "  inf" if mpg == float("inf") else "%5.0f" % mpg
        self.screen.blit(self.font.render("REAL-LIFE ESTIMATE (MPG)", True, C_ACCENT), (tx, ty))
        hdr = "%-6s %5s | %5s | %5s %5s %5s %5s" % (
            "speed", "aero", "noped", "p1", "p2", "p3", "p4")
        self.screen.blit(self.fsmall.render(hdr, True, C_TEXT_DIM), (tx, ty + 26))
        oy = ty + 44
        for v in MPG_SPEEDS_MPH:
            tot, aero, _ = road_load_watts(v, 1)
            row = "%-4dmph %4.0f%% | %s | %s %s %s %s" % (
                v, aero / max(1e-6, tot) * 100, cell(estimate_mpg(v, 1, False)),
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
            " a shape-optimized teardrop nearly HALVED the aero load, so",
            " 50 mph roughly DOUBLED: %.0f -> %.0f MPG (up to %.0f w/ 4 pedals)." % (
                242, estimate_mpg(50, 1, False), estimate_mpg(50, 4, True)),
            "",
            "AMBIENT HARVEST (~%.0f W, always on): solar film + EM suspension" % ambient_harvest_w(50),
            " dampers + triboelectric + tyre harvesters -- fuel-free, so below",
            " ~10-25 mph it ALONE covers the road load -> INFINITE MPG, no pedals.",
            "",
            "PEDALS (%.0f W/seat) stack on top of that:" % PEDAL_WATTS_PER_SEAT,
            " - more passengers = more free watts = infinite to a higher speed.",
            " - at high speed pedals are a rounding error, so they RETRACT into",
            "   the frame (useless against ~%.0f%% aero load at 80 mph)." % aero80,
            "",
            "80 mph is 'bad' only vs slower cruise -- ~%.0f MPG is still huge" % estimate_mpg(80, 1, False),
            "for a car, but a gentle cruise makes 5-15x that. Ease off = win.",
        ]
        oy += 8
        for ln in why:
            c = C_ACCENT if ln.endswith(":") else (
                C_GOOD if "INFINITE" in ln or "SLOWER" in ln else C_TEXT_DIM)
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
    print(" part work. Press TAB to DRIVE it like a car game. Press I for the")
    print(" full informational specification.")
    print("-" * 70)
    print(" Controls:")
    print("   TAB  switch PREVIEW <-> DRIVE    M  real-life MPG chart    I  info    H  help")
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
