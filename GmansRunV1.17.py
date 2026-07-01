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
     gear-increase core slips to a higher gear through the 2nd wet clutch, and
     the axial-flux generator + small angled "receival" output gear spin up.
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

    # --- 2nd wet clutch (core -> generator, "shift higher / slip harder") -
    "clutch2_d_mm":         260.0,
    "clutch2_thick_mm":     20.0,

    # --- Axial-flux generator ---------------------------------------------
    "gen_d_mm":             300.0,
    "gen_thick_mm":         110.0,

    # --- Toothed "receival" output gear (meshes trans teeth -> generator) --
    "out_gear_d_mm":        150.0,
    "out_gear_thick_mm":    55.0,
    "out_angle_deg":        14.0,    # the "unavoidable angle loss" pivot
}

# Firing / RPM behaviour
FIRING_IDLE_RPM   = 320.0     # always-spinning warm idle when vehicle moving
FIRING_GEN_RPM    = 3000.0    # generating RPM (chambers fire 1x / rev)
FLYWHEEL_MAX_RPM  = 20000.0   # magnetic-bearing safe ceiling
OMEGA_VIEW        = 2.513     # preview rotation rate at 1.0x (~2.5 s / rev, viewable)
INJECTOR_ANGLE    = math.pi / 2.0   # injector fixed at the top of the rim
MAX_PISTON_RPM    = 4500.0    # piston RPM at full power / 8 combustions per rev
VIEW_ROT_SCALE    = 0.03      # slows real RPMs to a viewable on-screen spin rate
OUTGEAR_RATIO     = DIMS["trans_outer_d_mm"] / DIMS["out_gear_d_mm"]  # trans teeth -> gear speed-up

# Thermal + heat-recovery (closed-loop water/ORC) model
THERM = {
    "ambient_c":   35.0,      # coolant inlet / ambient
    "mass":        9.0,       # engine thermal mass -> heats up FAST
    "comb_kw":     92.0,      # combustion power at full load / 8 combustions
    "heat_frac":   0.345,     # 1 - thermal_eff -> waste heat fraction
    "cool_k":      0.61,      # coolant heat-removal coefficient (per flow, per dT)
    "slip_k1":     11.0,      # main-clutch friction heat coefficient
    "slip_k2":     6.0,       # trans-clutch friction heat coefficient
    "orc_eff":     0.14,      # recovered heat -> electricity (ORC / steam)
    "steam_c":     80.0,      # min block temp for the ORC loop to make steam power
    "warn_c":      125.0,     # thermal warning threshold
}

# Vehicle / physics (lightweight series-hybrid laboratory mule)
VEH = {
    "curb_mass_kg":    950.0,   # ultralight carbon monocoque + driver
    "Cd":              0.13,
    "frontal_area_m2": 2.0,
    "Crr":             0.0038,
    "air_density":     1.225,
    "g":               9.81,
    "wheel_radius_m":  0.32,
    "drivetrain_eff":  0.93,    # motor + inverter chain
    "max_motor_kw":    240.0,   # 4 in-wheel axial-flux motors combined
    "max_brake_n":     9000.0,
}

# Electrical storage
ELEC = {
    "batt_kwh":        40.0,
    "soc_min":         0.18,    # never held high; max regen headroom
    "soc_max":         0.58,
    "soc_start":       0.45,
    "supercap_kwh":    0.55,    # 400+ Farad bank, real-time charge/drain
    "engine_elec_kw":  32.0,    # generator electric output when engine fires
    "thermal_eff":     0.655,   # 65.5% rotary thermal efficiency target
    "gasoline_kwh_gal": 33.7,   # energy content of a gallon of gasoline
}
ELEC["elec_kwh_per_gal"] = ELEC["gasoline_kwh_gal"] * ELEC["thermal_eff"]

# Engine control thresholds (keeps engine runtime < ~2% of drive time)
ENGINE_ON_SOC   = 0.22        # fire a burst when buffers run low
ENGINE_OFF_SOC  = 0.40        # stop the burst once topped to here


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
                 pivot=(0.0, 0.0, 0.0), tilt=(0.0, 0.0), hot=False):
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

    def world_verts(self, angle):
        v = self.verts
        if self.spin:
            v = v @ rot_z(angle * self.spin).T        # spin about the part's OWN axis
        rx, ry = self.tilt
        if rx or ry:
            v = v @ (rot_x(rx) @ rot_y(ry)).T         # then orient it (tilt into place)
        return v + self.pivot


# ---- primitive builders ----------------------------------------------------

def _annulus_cylinder(r_out, r_in, z0, z1, seg=44):
    """Hollow tube (ring) closed at both axial ends. Returns (verts, faces)."""
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


def _rotary_ring(zc):
    """One rotary combustion piston-ring + its 8 concave chambers at axial zc.
    Chamber meshes carry chamber_index 0..7 (shared firing schedule)."""
    mm = 0.001
    r_out = DIMS["ring_outer_d_mm"] * mm / 2
    r_in = DIMS["ring_inner_d_mm"] * mm / 2
    th = DIMS["ring_thickness_mm"] * mm
    cw = DIMS["chamber_width_mm"] * mm
    cd = DIMS["chamber_depth_mm"] * mm
    v, f = _annulus_cylinder(r_out, r_in, zc - th / 2, zc + th / 2, seg=60)
    meshes = [Mesh(v, f, C_RING, group="piston", hot=True)]
    for i in range(DIMS["chambers"]):
        a = i * 2 * math.pi / DIMS["chambers"]
        rr = r_out - cd / 2
        cx, cy = rr * math.cos(a), rr * math.sin(a)
        v, f = _box(cx, cy, zc, cd, cw, th * 0.8)
        v = (np.asarray(v) - np.array([cx, cy, zc])) @ rot_z(a).T + np.array([cx, cy, zc])
        m = Mesh(v, f, C_CHAMBER, group="piston", hot=True)
        m.chamber_index = i
        meshes.append(m)
        ax, ay = (r_out * 0.99) * math.cos(a), (r_out * 0.99) * math.sin(a)
        v, f = _box(ax, ay, zc, 0.012, cw, th * 0.85)
        v = (np.asarray(v) - np.array([ax, ay, zc])) @ rot_z(a).T + np.array([ax, ay, zc])
        meshes.append(Mesh(v, f, (150, 160, 175), group="piston", hot=True))
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


def _gear_teeth(radius, zc, count, half_th, col, pivot, tilt):
    """A ring of gear teeth in a part's LOCAL frame (pivot/tilt applied later)."""
    meshes = []
    for k in range(count):
        a = k * 2 * math.pi / count
        cx, cy = radius * math.cos(a), radius * math.sin(a)
        v, f = _box(cx, cy, zc, 0.016, 0.013, half_th * 2)
        v = (np.asarray(v) - np.array([cx, cy, zc])) @ rot_z(a).T + np.array([cx, cy, zc])
        meshes.append(Mesh(v, f, col, pivot=pivot, tilt=tilt))
    return meshes


def _integral_clutch(zc):
    """The MAIN wet clutch, integral to the piston and sitting in its inner bore:
    one face is the piston wall (spins with 'piston'), the mating face connects
    to the central 'core'. Full pressure locks them 1:1; slipping shows the two
    faces turning at different rates. One whole part -- not a plate stack."""
    mm = 0.001
    r_in = DIMS["ring_inner_d_mm"] * mm / 2
    meshes = []
    v, f = _annulus_cylinder(r_in * 0.96, r_in * 0.60, zc - 0.020, zc + 0.006, seg=36)
    meshes.append(Mesh(v, f, (95, 165, 190), group="piston"))     # piston-wall face
    v, f = _solid_cylinder(r_in * 0.58, zc - 0.006, zc + 0.022, seg=28)
    meshes.append(Mesh(v, f, (150, 150, 160), group="core"))      # core-connected face
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


def _cooling_recovery(zc, r_out, gx, gz):
    """Coolant manifold + pump + the closed-loop ORC heat-recovery unit (boiler,
    steam expander + mini generator, condenser) that turns engine heat into
    extra electricity. Sits below the engine (-Y). The mini generator = 'orc'."""
    meshes = []
    bx, by = 0.0, -(r_out + 0.34)          # ORC pack centre, below the engine
    # coolant manifold ring hugging the block OD (stationary)
    v, f = _annulus_cylinder(r_out * 1.10, r_out * 1.03, zc - 0.03, zc + 0.03, seg=40)
    meshes.append(Mesh(v, f, (45, 95, 150), group="static"))
    # coolant pump (top of engine)
    v, f = _solid_cylinder(0.05, zc - 0.03, zc + 0.03, seg=14)
    v = np.asarray(v) + np.array([0.0, r_out + 0.06, 0.0])
    meshes.append(Mesh(v, f, (60, 130, 190), group="orc"))
    # boiler / heat exchanger (finned, absorbs coolant heat) -- HOT
    v, f = _solid_cylinder(0.11, bx - 0.09, bx + 0.09, seg=20)
    v = np.asarray(v) @ rot_y(math.pi / 2).T + np.array([0.0, by, zc])
    meshes.append(Mesh(v, f, (170, 90, 60), group="static", hot=True))
    for k in range(12):                    # boiler fins
        a = k * 2 * math.pi / 12
        yy, zz = by + 0.12 * math.cos(a), zc + 0.12 * math.sin(a)
        v, f = _box(0.0, yy, zz, 0.16, 0.012, 0.03)
        meshes.append(Mesh(v, f, (150, 80, 55), group="static", hot=True))
    # steam expander + MINI GENERATOR (spins with recovered power)
    v, f = _solid_cylinder(0.07, -0.05, 0.05, seg=18)
    v = np.asarray(v) @ rot_y(math.pi / 2).T + np.array([0.20, by, zc])
    meshes.append(Mesh(v, f, (90, 120, 180), group="orc"))
    v, f = _solid_cylinder(0.055, -0.05, 0.05, seg=18)
    v = np.asarray(v) @ rot_y(math.pi / 2).T + np.array([0.30, by, zc])
    meshes.append(Mesh(v, f, (70, 150, 210), group="orc"))
    # condenser (cool) + pipes
    v, f = _box(-0.24, by, zc, 0.10, 0.16, 0.10)
    meshes.append(Mesh(v, f, (50, 110, 165), group="static"))
    meshes.append(_pipe((0.0, r_out + 0.06, zc), (0.0, by + 0.12, zc), 0.014, (60, 130, 190)))
    meshes.append(_pipe((0.09, by, zc), (0.16, by, zc), 0.012, (200, 120, 90)))
    meshes.append(_pipe((-0.11, by, zc), (-0.20, by, zc), 0.012, (70, 150, 210)))
    return meshes


def _hydraulic(zc, r_out):
    """Hydraulic power pack: pump + accumulator + valve block + pressure lines to
    BOTH wet clutches (they engage on hydraulic fluid pressure). Sits at -X."""
    meshes = []
    hx = -(r_out + 0.30)
    v, f = _box(hx, 0.0, zc, 0.12, 0.16, 0.12)          # valve block
    meshes.append(Mesh(v, f, (150, 120, 60), group="static"))
    v, f = _solid_cylinder(0.06, -0.05, 0.05, seg=16)   # pump
    v = np.asarray(v) @ rot_y(math.pi / 2).T + np.array([hx + 0.12, 0.06, zc])
    meshes.append(Mesh(v, f, (190, 150, 70), group="orc"))
    v, f = _solid_cylinder(0.07, zc - 0.10, zc + 0.10, seg=16)   # accumulator
    v = np.asarray(v) + np.array([hx, -0.14, 0.0])
    meshes.append(Mesh(v, f, (120, 100, 55), group="static"))
    # pressure lines fanning toward the clutch stacks (front, +Z)
    meshes.append(_pipe((hx + 0.06, 0.0, zc), (hx + 0.30, 0.0, zc + 0.2), 0.011, (210, 170, 80)))
    meshes.append(_pipe((hx + 0.06, 0.0, zc), (hx + 0.30, 0.0, zc - 0.2), 0.011, (210, 170, 80)))
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
    # The rings BIND together to engage a wider effective disc = higher gear;
    # the ring gears carry TEETH on the OD for output. No separate 2nd clutch.
    n = DIMS["trans_rings"]
    ro = DIMS["trans_outer_d_mm"] * mm / 2
    ri = DIMS["trans_inner_d_mm"] * mm / 2
    tth = DIMS["trans_thick_mm"] * mm
    zt = -th / 2 - tth / 2 - 0.02
    tmeshes = []
    for k in range(n):
        outer = ri + (ro - ri) * (n - k) / n
        inner = ri + (ro - ri) * (n - k - 1) / n
        col = C_TRANS if k % 2 == 0 else C_TRANS_ALT
        v, f = _annulus_cylinder(outer * 0.985, inner, zt - tth / 2, zt + tth / 2, seg=48)
        tmeshes.append(Mesh(v, f, col, group="trans%d" % k))       # each ring its rate
    # output teeth around the OD, turning at the bound transmission-output rate
    tmeshes += _grp(_gear_teeth(ro, zt, 30, tth / 2, (185, 150, 55), (0, 0, 0), (0, 0)), "transout")
    parts.append(Part("trans", "Transmission Ring-Layers (+ output teeth)", tmeshes,
        ["Function: %d concentric RING LAYERS that BIND together to engage --" % n,
         "the binding IS the 'clutch': more bind = a wider effective disc = a",
         "higher gear; partial bind = slippable ratio, full bind = locked gear.",
         "The ring gears carry TEETH on the OD that drive the output gear.",
         "Rings, not discs. Ratios ~1.00:1 -> ~2.60:1. No separate 2nd clutch."],
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
         "Dia %.0f mm x %.0f mm axial-flux PM, ~%.0f kW when firing" % (
             DIMS["gen_d_mm"], DIMS["gen_thick_mm"], ELEC["engine_elec_kw"])],
        order=nextord(), explode=(0, 0, -1.05), color=C_GEN))

    # --- Toothed receival output gear (meshes trans teeth -> generator) --
    rog = DIMS["out_gear_d_mm"] * mm / 2
    oth = DIMS["out_gear_thick_mm"] * mm
    ang = math.radians(DIMS["out_angle_deg"])
    px = ro + rog - 0.012
    v, f = _solid_cylinder(rog, -oth / 2, oth / 2, seg=28)
    ogm = [Mesh(v, f, C_OUTGEAR, group="out", pivot=(px, 0.0, zt), tilt=(ang, 0.0))]
    ogm += _grp(_gear_teeth(rog, 0.0, 18, oth / 2, (185, 150, 55), (px, 0.0, zt), (ang, 0.0)), "out")
    parts.append(Part("outgear", "Output Receival Gear (-> generator)", ogm,
        ["Function: meshes the transmission ring TEETH and drives the generator",
         "Dia %.0f mm; rotates on its OWN axis like a gear, ~%.0f deg pivot" % (
             DIMS["out_gear_d_mm"], DIMS["out_angle_deg"]),
         "Unavoidable pivot angle loss taking power off the toothed rim",
         "Feeds the generator to CHARGE batteries/caps -- not the wheels."],
        order=nextord(), explode=(0.95, 0.0, -0.72), color=C_OUTGEAR))

    # --- N rotary pistons, each with an INTEGRAL main clutch + injector -
    for k in range(n_pistons):
        zk = k * pitch
        sfx = "" if n_pistons == 1 else " #%d" % (k + 1)
        parts.append(Part("ring%d" % k, "Rotary Piston%s (8-chamber)" % sfx, _rotary_ring(zk),
            ["Function: circular 'water-wheel' rotary piston on the shared core",
             "OD %.0f / ID %.0f mm, thick %.0f mm  (very thin)" % (
                 DIMS["ring_outer_d_mm"], DIMS["ring_inner_d_mm"], DIMS["ring_thickness_mm"]),
             "%d concave paddles @45 deg; each combusts as it passes the injector" % DIMS["chambers"],
             "1 to 8 combustions per single rotation, on demand",
             "At full lock in high gear, the rotation rate is the limit"],
            order=nextord(), explode=(0, 0, 0.42 + k * 0.42), color=C_RING))

        parts.append(Part("clutch1_%d" % k, "Main Wet Clutch%s (in-piston)" % sfx, _integral_clutch(zk),
            ["Function: INTEGRAL to the piston -- one face IS the piston wall,",
             "the mating face connects straight to the central Core.",
             "Sits in the piston's inner bore; hydraulic pressure engages it.",
             "FULL engagement = 1:1 lock, grabs full piston power (no slip);",
             "slipping shows the two faces turning at different rates."],
            order=nextord(), explode=(0, 0, 0.28 + k * 0.42), color=C_CLUTCH))

        parts.append(Part("inj%d" % k, "Fuel Injector%s (direct)" % sfx, _injector(zk),
            ["Function: fixed direct-injection nozzle at the rim top",
             "Each chamber injects + combusts as it sweeps past, like a water wheel",
             "Idle = 1 combustion/rev; full power = 8 combustions/rev",
             "Lean direct injection synced to rotation (to scale)"],
            order=nextord(), explode=(0.0, 0.55, 0.42 + k * 0.42), color=(230, 200, 90)))

    # --- Hydraulic power pack (engages both wet clutches) --------------
    parts.append(Part("hyd", "Hydraulic Power Pack (clutch actuation)", _hydraulic(0.0, r_out),
        ["Function: pressurizes the hydraulic fluid that ENGAGES both clutches",
         "Pump + accumulator + valve block + pressure lines to each clutch",
         "More pressure = harder engagement / less slip; low = free slip",
         "Main clutch fully engaged grabs full piston power (no slip)"],
        order=nextord(), explode=(-1.2, 0.0, 0.0), color=(150, 120, 60)))

    # --- Cooling loop + ORC heat-recovery mini generator --------------
    parts.append(Part("cool", "Cooling + Heat-Recovery ORC", _cooling_recovery(0.0, r_out, 0, 0),
        ["Function: closed water loop cools the block, then RECOVERS that heat",
         "Bored cooling tunnels in the block -> coolant -> boiler makes steam",
         "Steam expander spins a MINI GENERATOR (2nd source of electricity)",
         "Both generators (main + heat-driven) power the car; even coasting slip",
         "makes heat that is recycled -- cool water, steam, more electric."],
        order=nextord(), explode=(0.0, -1.2, 0.0), color=(60, 130, 190)))

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
    assembly views, mouse hover-picking, highlight + hover-pop, and the assembly
    'puzzle' (place parts one at a time)."""

    def __init__(self):
        self.n_pistons = 1
        self.parts = build_engine_parts(self.n_pistons)
        self.az = 0.65
        self.el = 0.50
        self.dist = 1.7
        self.pan = np.array([0.0, 0.0])
        self.light = np.array([0.4, 0.6, 1.0])
        self.light = self.light / np.linalg.norm(self.light)
        self.view = "full"                  # full | exploded | assembly
        self.explode_amt = 0.0              # animated 0..1
        self.assembled = len(self.parts)    # parts placed in assembly mode
        self.hovered = None
        self.selected = None
        self.pop = np.zeros(len(self.parts))
        self.hover_spread = 0.0

    def set_pistons(self, n):
        n = max(1, min(4, n))
        if n == self.n_pistons:
            return
        self.n_pistons = n
        self.parts = build_engine_parts(n)
        self.pop = np.zeros(len(self.parts))
        self.hovered = None
        self.selected = None
        self.assembled = len(self.parts)

    def reset_view(self):
        self.az, self.el, self.dist = 0.65, 0.50, 1.7
        self.pan = np.array([0.0, 0.0])

    def set_view(self, mode):
        self.view = mode
        if mode == "assembly" and self.assembled >= len(self.parts):
            self.assembled = 0
        self.selected = None

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
        sp_target = 0.55 if (hi is not None and self.view == "full") else 0.0
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
               view_override=None, injection=None, heat=0.0):
        clip = surf.get_clip()
        surf.set_clip(rect)
        cx = rect.x + rect.w / 2.0 + self.pan[0]
        cy = rect.y + rect.h / 2.0 + self.pan[1]
        focal = min(rect.w, rect.h) * 0.95
        Rcam = rot_x(self.el) @ rot_y(self.az)
        default_ang = angles.get("default", 0.0)

        vw = view_override or self.view
        eamt = 0.0 if view_override == "full" else self.explode_amt
        if view_override is None and self.view == "full":
            eamt += self.hover_spread        # open up the engine on hover
        interactive = interactive and view_override is None
        # crisp facet outlines in the main preview at <=2 pistons; off for the
        # small drive inset and heavy 3-4 piston configs (keeps it fast)
        do_outline = (view_override is None) and (self.n_pistons <= 2)
        hi = (self.selected if self.selected is not None else self.hovered)
        if view_override is not None:
            hi = None

        polys = []
        labels = []
        leaders = []
        screeninfo = []
        chamber_fx = []
        lx, ly, lz = float(self.light[0]), float(self.light[1]), float(self.light[2])

        for pi, part in enumerate(self.parts):
            base_off, dim, tag = self._layout(pi, vw, eamt)
            pop = self.pop[pi] if view_override is None else 0.0
            off = base_off + part.popdir * (pop * 0.26)
            highlight = (pi == hi)
            allcam = []
            for m in part.meshes:
                wv = m.world_verts(angles.get(m.group, default_ang)) + off
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
                if (m.chamber_index is not None and tag != "pending"
                        and min(dzl) > 0.05):
                    nv = len(sxl)
                    mcx = sum(sxl) / nv
                    mcy = sum(syl) / nv
                    crad = max(math.hypot(sxl[i] - mcx, syl[i] - mcy) for i in range(nv))
                    chamber_fx.append((cam[:, 2].mean(), mcx, mcy, max(6.0, crad),
                                       m.chamber_index))
                for face in m.faces:
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

        # fuel-injection + combustion flashes, drawn over the chambers
        # (far-to-near so nearer pistons sit on top)
        chamber_fx.sort(key=lambda t: t[0], reverse=True)
        for _, mcx, mcy, crad, ci in chamber_fx:
            inj = injection[ci] if injection is not None else 0.0
            glow = firing_glow[ci]
            if inj > 0.05:
                draw_injection(surf, mcx, mcy, crad, inj)
            if glow > 0.05:
                draw_combustion(surf, mcx, mcy, crad, glow)

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
        self.ring_rpm = 0.0
        self.clutch1_engage = 0.0
        self.clutch2_engage = 0.0
        self.fuel_gal = 0.0
        self.miles = 0.0
        self.engine_seconds = 0.0
        self.total_seconds = 0.0
        self.firing_phase = 0.0
        self.firing_glow = np.zeros(DIMS["chambers"])
        self.regen_active = False
        self.last_mpg = 0.0
        self.flow = {"engine": 0.0, "regen": 0.0, "trac": 0.0, "fly": 0.0}

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
        """Animate the mechanical drivetrain state (ring RPM + both clutches)."""
        target_rpm = (FIRING_GEN_RPM if firing
                      else (FIRING_IDLE_RPM if moving else 0.0))
        self.ring_rpm += (target_rpm - self.ring_rpm) * min(1.0, dt * 2.5)
        tgt_c1 = 1.0 if firing else (0.25 if moving else 0.0)
        self.clutch1_engage += (tgt_c1 - self.clutch1_engage) * min(1.0, dt * 3)
        self.clutch2_engage += ((0.9 if firing else 0.15)
                                - self.clutch2_engage) * min(1.0, dt * 3)

    def update_demo(self, dt):
        """PREVIEW mode: always-firing showcase. Spins the ring and winds the
        flywheel up for display without touching the drive economy (fuel/miles)."""
        self.engine_on = True
        self._mech(dt, True, True)
        target = 9000.0 * 2 * math.pi / 60.0     # gentle visual wind-up
        self.fly_omega += (target - self.fly_omega) * min(1.0, dt * 0.15)

    def update(self, dt, v_mph, road_power_kw, moving):
        """road_power_kw: + = power demanded at the wheels, - = power available
        from braking / downhill (regen)."""
        self.total_seconds += dt
        self.miles += v_mph * (dt / 3600.0)

        # engine burst controller (runs only when buffers low, or when forced)
        if self.force_timer > 0.0:
            self.force_timer = max(0.0, self.force_timer - dt)
            self.engine_on = True
        elif not self.engine_on and self.soc < ENGINE_ON_SOC:
            self.engine_on = True
        elif self.engine_on and self.soc >= ENGINE_OFF_SOC:
            self.engine_on = False

        # rotary ring RPM + clutch engagement
        self._mech(dt, self.engine_on, moving)

        # engine electrical production + fuel burn
        engine_kw = ELEC["engine_elec_kw"] if self.engine_on else 0.0
        if self.engine_on:
            self.engine_seconds += dt
            self.fuel_gal += engine_kw * (dt / 3600.0) / ELEC["elec_kwh_per_gal"]
        self.flow["engine"] = engine_kw

        trac_kw = max(0.0, road_power_kw) / VEH["drivetrain_eff"]
        regen_kw = max(0.0, -road_power_kw) * 0.85
        self.flow["trac"] = trac_kw
        self.flow["regen"] = regen_kw

        net_kw = engine_kw + regen_kw - trac_kw
        dE = net_kw * (dt / 3600.0)                  # kWh this step

        # route energy: supercap first, then battery, then flywheel
        self.cap_kwh += dE
        fly_dE = 0.0
        if self.cap_kwh > self.cap_max:
            spill = self.cap_kwh - self.cap_max
            self.cap_kwh = self.cap_max
            room = (ELEC["soc_max"] - self.soc) * self.batt_kwh
            into_batt = min(spill, max(0.0, room))
            self.soc += into_batt / self.batt_kwh
            fly_dE = spill - into_batt               # momentum harvest -> flywheel
        elif self.cap_kwh < 0.0:
            need = -self.cap_kwh
            self.cap_kwh = 0.0
            avail = max(0.0, self.soc * self.batt_kwh)
            from_batt = min(need, avail)
            self.soc -= from_batt / self.batt_kwh
            fly_dE = -(need - from_batt)             # drain flywheel for the rest

        if fly_dE != 0.0:
            self.flow["fly"] = fly_dE / (dt / 3600.0)
            E = max(0.0, self.fly_kwh + fly_dE) * 3.6e6
            self.fly_omega = math.sqrt(max(0.0, 2 * E / self.fly_I))
            self.fly_omega = min(self.fly_omega,
                                 FLYWHEEL_MAX_RPM * 2 * math.pi / 60.0)
        else:
            self.flow["fly"] *= 0.9

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

    def update(self, dt, throttle, brake, pt, force_grade=0.0):
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
        pt.update(dt, abs(v_mph), road_power_kw, moving=abs(self.v) > 0.3)
        return grade, v_mph


# =============================================================================
# SECTION 6 -- FULL INFORMATIONAL SPECIFICATION
# =============================================================================

INFO_SECTIONS = [
    ("HOHEV-ROTARY GEN 4  --  CONCEPT OVERVIEW", [
        "A series-hybrid EV whose only fuel burner is an 8-chamber CIRCULAR",
        "rotary 'water-wheel' combustion ring. It runs in short bursts only to",
        "make electricity, drives an axial-flux generator at ~1:1, and otherwise",
        "the car runs electric -- harvesting momentum on every coast, brake and",
        "downhill. Concept target: 1,000-1,800 MPG under ideal conditions.",
    ]),
    ("1. ROTARY PISTON(S) + FIRING MODEL (the circular 'piston')", [
        "Outer dia %.0f mm | inner hollow %.0f mm | thick %.0f mm (very thin)." % (
            DIMS['ring_outer_d_mm'], DIMS['ring_inner_d_mm'], DIMS['ring_thickness_mm']),
        "%d concave paddles every 45 deg, like a water wheel. A FIXED injector at" % DIMS['chambers'],
        "the rim top direct-injects each chamber as it sweeps past; that chamber",
        "then combusts -- injection, combustion, torque, repeat, in sequence.",
        "Combust 1 to 8 times per SINGLE rotation on demand: idle = 1/rev, full",
        "power = 8/rev. At full lock in high gear the ROTATION RATE is the limit.",
        "Run 1-4 stacked pistons on the same core for more torque (preview [ ]).",
        "FIRING MODE (key M): SEQUENTIAL (one chamber at a time, water-wheel) or",
        "SIMULTANEOUS (all scheduled chambers fire together = one synergetic pulse).",
    ]),
    ("2. MAIN WET CLUTCH -- INTEGRAL TO THE PISTON", [
        "The main clutch is BUILT INTO the piston, in its inner bore: one face IS",
        "the piston wall, the mating face connects straight to the central Core.",
        "Hydraulic fluid pressure engages it (shared pump + accumulator + valves).",
        "FULL engagement = 1:1 lock, grabs FULL piston power (no slip); slipping",
        "shows the two faces turning at different rates. One clutch per piston,",
        "all engaging the SAME core. (One whole part -- not a plate stack.)",
    ]),
    ("2b. COOLING TUNNELS + HEAT-RECOVERY (ORC)", [
        "The block runs hot fast -- combustion heat PLUS clutch-slip friction heat.",
        "Bored cooling TUNNELS carry a closed water loop through the block. That",
        "removed heat boils water to steam, spins a MINI GENERATOR (Organic",
        "Rankine Cycle) and recovers extra electricity. So TWO generators drive",
        "the car: the main rotary-driven one AND the heat-driven one. Even coasting",
        "downhill, clutch/brake drag makes heat that is recycled -- cool water,",
        "steam, more electric. Manage it with the Coolant Flow slider.",
    ]),
    ("3. TUNGSTEN KINETIC FLYWHEEL -- ON THE CORE", [
        "Weighted disc, %.0f mm, %.0f kg tungsten, mounted ON the central core." % (
            DIMS['flywheel_d_mm'], DIMS['flywheel_mass_kg']),
        "When the MAIN CLUTCH disengages the disc FREE-SPINS with its stored",
        "energy -- the transmission can take that energy back, or use it to",
        "reduce stall. Wound up by downhill / coast momentum while the engine is",
        "OFF. Magnetic bearings, <0.3%% drag, safe to %.0f RPM." % FLYWHEEL_MAX_RPM,
    ]),
    ("4. TRANSMISSION = RING LAYERS (the rings ARE the 'clutch')", [
        "%d concentric RING LAYERS -- not discs. The rings BIND together to" % DIMS['trans_rings'],
        "engage: more bind = a wider effective disc = a higher gear. Partial bind",
        "= a slippable ratio; full bind = a locked fixed gear. That binding IS the",
        "engagement -- there is NO separate 2nd wet clutch. The ring gears carry",
        "TEETH on the OD for output. Lives inside the hollow (thin engine).",
        "Ratios ~1.00:1 -> ~2.60:1.",
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
        "Routing priority: engine/regen -> supercap -> battery -> flywheel.",
        "Engine fires only below %.0f%% SOC, stops at %.0f%%. Goal: engine runtime" % (
            ENGINE_ON_SOC * 100, ENGINE_OFF_SOC * 100),
        "under ~2%% of drive time. Never chase a full battery -- drain to the next",
        "downhill and harvest it back.",
    ]),
    ("8. VEHICLE (lightweight series-hybrid mule)", [
        "Curb ~%.0f kg | Cd %.2f | frontal %.1f m^2 | Crr %.4f." % (
            VEH['curb_mass_kg'], VEH['Cd'], VEH['frontal_area_m2'], VEH['Crr']),
        "Four in-wheel axial-flux motors, ~%.0f kW combined." % VEH['max_motor_kw'],
        "Magnetic motor decoupling for true zero-drag coasting.",
    ]),
    ("9. MANUAL ENGINE TEST BENCH (preview, drag sliders)", [
        "Every input is manual so the engine can be tested. Drag the left-panel",
        "sliders: ENGINE POWER, HYDRAULIC PRESSURE, MAIN CLUTCH, TRANSMISSION BIND,",
        "COOLANT FLOW and GEAR. The clutch only bites with hydraulic pressure.",
        "Live kinematics: piston -> main clutch (slip) -> core -> transmission ring",
        "bind (blends 1:1 up to the gear ratio) -> toothed output gear -> generator.",
        "Every part/tooth/screw turns at its TRUE RPM (slipping faces + ring layers",
        "spin at different rates). Readouts show RPM, ratio, slip %, temp, recovery.",
        "show RPM, ratio, slip %, BLOCK TEMP, heat removed and main+ORC electric.",
    ]),
    ("10. WATCHING IT WORK (playback)", [
        "The preview always shows fuel injection, combustion and rotation to scale.",
        "P pauses; - / = step the view speed (down to 0.05x) so you can watch a",
        "single chamber inject then combust frame by frame. , / . set combustions",
        "per rev (1-8); M toggles sequential vs simultaneous; [ / ] add pistons.",
    ]),
    ("CONTROLS", [
        "TAB ........ switch PREVIEW <-> DRIVE          I ... info     H ... help",
        "PREVIEW VIEWS:  1 = FULL   2 = EXPLODED   3 = ASSEMBLY (puzzle)",
        "PREVIEW: drag orbit | wheel zoom | right-drag pan | L labels | R reset",
        "         hover a part = read spec card | click = pin / unpin selection",
        "TEST BENCH: drag the left sliders (power, both clutches, gear)",
        "PLAYBACK: P pause | - / = view speed | [ ] pistons | , . combustions/rev",
        "          M = firing mode (sequential <-> simultaneous)",
        "ASSEMBLY: click or N = place next part | B = back | F = all | 0 = clear",
        "DRIVE:  W/Up throttle | S/Down brake | A/D steer | C cruise | E engine burst",
        "        R = shift Drive <-> Reverse (when stopped) | G downhill | SPACE brake",
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
        self.W, self.H = 1280, 760
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
        self.combustions = 8                       # combustions per revolution
        self.fire_mode = "SEQUENTIAL"              # SEQUENTIAL | SIMULTANEOUS
        self._injection = np.zeros(DIMS["chambers"])
        # --- manual engine test-bench inputs (all mouse-adjustable) --------
        self.engine_power = 0.6                    # 0..1 piston drive
        self.hyd_pressure = 1.0                    # hydraulic supply pressure
        self.clutch_main = 1.0                     # 1st (main) wet clutch command
        self.clutch_trans = 1.0                    # 2nd (transmission) wet clutch
        self.coolant_flow = 0.7                    # coolant pump flow
        self.gear_sel = 0                          # selected transmission ring
        self.piston_rpm = 0.0
        self.fly_rpm_v = 0.0
        self.group_angle = {"default": 0.0}        # live angle per kinematic group
        self.kin = {"piston": 0, "core": 0, "gen": 0, "out": 0, "trans_out": 0,
                    "ratio": 1.0, "gear": 1, "ngear": DIMS["trans_rings"], "gr": 1.0,
                    "slip1": 0, "slip2": 0, "fly": 0, "ep": 0, "c1": 0, "c2": 0}
        # thermal + heat-recovery state
        self.temp_c = THERM["ambient_c"]
        self.therm = {"temp": self.temp_c, "heat_kw": 0.0, "cool_kw": 0.0,
                      "orc_kw": 0.0, "main_kw": 0.0, "total_kw": 0.0}
        self._drag_slider = None
        self.show_info = False
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
                self.W, self.H = max(960, e.w), max(620, e.h)
                self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
                self._rebuild_bg()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_TAB:
                    self.mode = 1 - self.mode
                elif e.key == pygame.K_i:
                    self.show_info = not self.show_info
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
                    self.combustions = {8: 4, 4: 2, 2: 1, 1: 1}[self.combustions]
                elif e.key == pygame.K_PERIOD and self.mode == self.MODE_PREVIEW:
                    self.combustions = {1: 2, 2: 4, 4: 8, 8: 8}[self.combustions]
                elif e.key == pygame.K_m and self.mode == self.MODE_PREVIEW:
                    self.fire_mode = ("SIMULTANEOUS" if self.fire_mode == "SEQUENTIAL"
                                      else "SEQUENTIAL")
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
                    hit = self._bench_hit(e.pos) if self.mode == self.MODE_PREVIEW else None
                    if hit is not None:
                        self._drag_slider = hit
                        self._bench_set_from_x(hit, e.pos[0])
                    else:
                        self.dragging = True
                        self._press_pos = e.pos
                elif e.button == 3:
                    self.panning = True
                elif e.button in (4, 5) and self.mode == self.MODE_PREVIEW:
                    self.renderer.dist *= 0.9 if e.button == 4 else 1.1
                    self.renderer.dist = max(0.8, min(6.0, self.renderer.dist))
            elif e.type == pygame.MOUSEBUTTONUP:
                if (e.button == 1 and self._drag_slider is None and self._press_pos is not None
                        and self.mode == self.MODE_PREVIEW and not self.show_info):
                    moved = math.hypot(e.pos[0] - self._press_pos[0],
                                       e.pos[1] - self._press_pos[1])
                    if moved < 6:
                        self._preview_click()
                self._press_pos = None
                self._drag_slider = None
                self.dragging = self.panning = False
            elif e.type == pygame.MOUSEMOTION:
                dx, dy = e.rel
                if self._drag_slider is not None:
                    self._bench_set_from_x(self._drag_slider, e.pos[0])
                elif self.mode == self.MODE_PREVIEW and not self.show_info:
                    if self.dragging:
                        self.renderer.az += dx * 0.01
                        self.renderer.el += dy * 0.01
                        self.renderer.el = max(-1.5, min(1.5, self.renderer.el))
                    elif self.panning:
                        self.renderer.pan += np.array([dx, dy])
            elif e.type == pygame.MOUSEWHEEL and self.show_info:
                self.info_scroll = max(0, self.info_scroll - e.y * 30)

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
        SEQUENTIAL: each scheduled chamber injects as it nears the rim injector
        then combusts as it sweeps past (water-wheel). SIMULTANEOUS: all the
        scheduled chambers fire together in one synergetic pulse per rev."""
        glow = self.pt.firing_glow
        inj = self._injection
        glow[:] = 0.0
        inj[:] = 0.0
        if not active:
            return
        nch = DIMS["chambers"]
        step = nch // combustions                 # 1->all, 8->only one

        def window(d):
            # wider trailing glow so several chambers read as "firing" at once
            comb = max(0.0, 1.0 - d / 1.5) if -0.06 < d < 1.5 else 0.0
            ij = max(0.0, 1.0 - (abs(d) - 0.06) / 0.54) if -0.6 < d <= -0.06 else 0.0
            return comb, ij

        if mode == "SIMULTANEOUS":
            # one synchronized pulse when chamber 0 reaches the injector
            d0 = ((self.engine_angle - INJECTOR_ANGLE + math.pi) % (2 * math.pi)) - math.pi
            comb, ij = window(d0)
            for i in range(0, nch, step):
                glow[i] = comb
                inj[i] = ij
        else:
            for i in range(0, nch, step):
                base = i * 2 * math.pi / nch
                d = ((self.engine_angle + base - INJECTOR_ANGLE + math.pi)
                     % (2 * math.pi)) - math.pi    # 0 exactly at the injector
                glow[i], inj[i] = window(d)

    def _kinematics(self, dt, drive):
        """Live drivetrain kinematics: piston -> main clutch (slip) -> core ->
        transmission ring (gear) -> trans clutch (slip) -> generator -> output.
        Advances every group's angle by its true RPM, and records readouts."""
        nrings = DIMS["trans_rings"]
        ratios = [1.0 + 1.6 * (k / max(1, nrings - 1)) for k in range(nrings)]
        if drive:
            moving = abs(self.world.v) > 0.3
            ep = 1.0 if self.pt.engine_on else (0.35 if moving else 0.12)
            c1 = 1.0 if self.pt.engine_on else (0.4 if moving else 0.0)
            c2 = self.pt.clutch2_engage
            gear = min(nrings - 1, int(abs(self.world.v) / 6.0))
        else:
            ep = self.engine_power
            # clutches engage ONLY via hydraulic pressure (command x supply)
            c1 = self.clutch_main * self.hyd_pressure
            c2 = self.clutch_trans * self.hyd_pressure
            gear = max(0, min(nrings - 1, int(self.gear_sel)))

        target = ep * (0.35 + 0.65 * self.combustions / 8.0) * MAX_PISTON_RPM
        self.piston_rpm += (target - self.piston_rpm) * min(1.0, dt * 1.5)
        core = self.piston_rpm * c1                 # main clutch slip (piston->core)
        gr = ratios[gear]
        eff = 1.0 + (gr - 1.0) * c2                 # ring-bind blends 1:1 -> gear
        trans_out = core * eff                      # bound transmission output
        gen = trans_out * OUTGEAR_RATIO             # output gear / generator speed
        # flywheel is on the core: couples when the main clutch is engaged,
        # otherwise FREE-SPINS with stored energy (slow bearing decay)
        if c1 > 0.05:
            self.fly_rpm_v += (core - self.fly_rpm_v) * min(1.0, dt * 0.4)
        else:
            self.fly_rpm_v *= max(0.0, 1.0 - dt * 0.05)
            core = max(core, self.fly_rpm_v * 0.15)  # flywheel can reduce stall

        self.kin = dict(piston=self.piston_rpm, core=core, gen=gen, out=gen,
                        trans_out=trans_out, ratio=eff, gear=gear + 1, ngear=nrings,
                        gr=gr, slip1=(1.0 - c1) * 100.0, slip2=(1.0 - c2) * 100.0,
                        fly=self.fly_rpm_v, ep=ep, c1=c1, c2=c2)

        rpm = {"piston": self.piston_rpm, "core": core, "fly": self.fly_rpm_v,
               "transout": trans_out, "gen": gen, "out": gen, "static": 0.0}
        for k in range(nrings):
            rpm["trans%d" % k] = core * ratios[k]
        ts = 1.0 if drive else (0.0 if self.paused else self.speeds[self.speed_idx])
        for grp, r in rpm.items():
            self.group_angle[grp] = (self.group_angle.get(grp, 0.0)
                                     + (r / 60.0 * 2 * math.pi) * dt * ts * VIEW_ROT_SCALE)
        self.group_angle["default"] = self.group_angle["piston"]
        self.engine_angle = self.group_angle["piston"]

    def _thermal(self, dt, drive):
        """Engine heat + closed-loop water/ORC recovery. Combustion + clutch slip
        heat the block; coolant flow removes heat; above the steam point that
        removed heat drives the ORC mini-generator, recovering extra electricity.
        Returns kW recovered (added to the drive's charging in DRIVE mode)."""
        k = self.kin
        if drive:
            flow = 0.75
            comb_active = 1.0 if self.pt.engine_on else 0.0
            slip_load = self.brake + (0.3 if abs(self.world.v) > 1 else 0.0)  # coasting drag
        else:
            flow = self.coolant_flow
            comb_active = 1.0 if k["ep"] > 0.02 else 0.0
            slip_load = 1.0

        comb_heat = comb_active * k["ep"] * (0.35 + 0.65 * self.combustions / 8.0) \
            * THERM["comb_kw"] * THERM["heat_frac"]
        slip1 = THERM["slip_k1"] * (k["piston"] / 1000.0) * (1.0 - k["c1"]) * max(k["c1"], 0.05) * slip_load
        slip2 = THERM["slip_k2"] * (k["core"] / 1000.0) * (1.0 - k["c2"]) * max(k["c2"], 0.05) * slip_load
        heat_in = comb_heat + slip1 + slip2

        cool = THERM["cool_k"] * flow * max(0.0, self.temp_c - THERM["ambient_c"])
        self.temp_c += (heat_in - cool) / THERM["mass"] * dt
        self.temp_c = max(THERM["ambient_c"], min(220.0, self.temp_c))

        orc = 0.0
        if self.temp_c > THERM["steam_c"] and flow > 0.05:
            usable = cool * min(1.0, (self.temp_c - THERM["steam_c"]) / 40.0)
            orc = usable * THERM["orc_eff"]
        main_kw = comb_active * k["ep"] * (0.35 + 0.65 * self.combustions / 8.0) \
            * THERM["comb_kw"] * (1.0 - THERM["heat_frac"])
        self.therm = {"temp": self.temp_c, "heat_kw": heat_in, "cool_kw": cool,
                      "orc_kw": orc, "main_kw": main_kw, "total_kw": main_kw + orc}
        return orc

    # ---------------------------------------------------------------- update
    def update(self, dt):
        self.renderer.tick(dt)
        drive = (self.mode == self.MODE_DRIVE)
        if drive:
            grade, v_mph = self.world.update(dt, self.throttle, self.brake,
                                             self.pt, self.force_grade)
            self.cur_grade = grade
            self.cur_mph = abs(v_mph)
        else:
            self.cur_grade = 0.0
            self.cur_mph = 0.0
        self._kinematics(dt, drive)
        orc = self._thermal(dt, drive)
        if drive:
            # recovered heat electricity trickles into the buffer (helps MPG)
            self.pt.cap_kwh = min(self.pt.cap_max, self.pt.cap_kwh + orc * (dt / 3600.0))
            self.pt.flow["orc"] = orc
            comb = 8 if (self.pt.flow["engine"] > 0 and self.throttle > 0.4) else max(1, self.combustions)
            self._compute_firing(self.pt.engine_on, comb, self.fire_mode)
        else:
            self._compute_firing(self.engine_power > 0.02, self.combustions, self.fire_mode)

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
        self.draw_topbar()
        pygame.display.flip()

    def draw_topbar(self):
        panel(self.screen, 0, 0, self.W, 34, alpha=190)
        if self.mode == self.MODE_PREVIEW:
            vname = {"full": "FULL", "exploded": "EXPLODED", "assembly": "ASSEMBLY"}
            mode = "PREVIEW [%s]" % vname[self.renderer.view]
        else:
            mode = "DRIVE  (simulation)"
        self.screen.blit(self.fbig.render("GmansRun V1.17", True, C_ACCENT), (12, 2))
        t = self.font.render("MODE: %s   [TAB] switch   [I] info   [H] help" % mode,
                             True, C_TEXT)
        self.screen.blit(t, (self.W - t.get_width() - 12, 9))

    # ---- preview ----
    def _heat_norm(self):
        return max(0.0, min(1.0, (self.temp_c - THERM["ambient_c"])
                            / (THERM["warn_c"] - THERM["ambient_c"])))

    def draw_preview(self):
        rect = pygame.Rect(0, 34, self.W, self.H - 34)
        self.renderer.render(self.screen, rect, self.group_angle, self.pt.firing_glow,
                             mouse_pos=pygame.mouse.get_pos(),
                             show_labels=self.show_labels, label_font=self.fsmall,
                             interactive=True, injection=self._injection,
                             heat=self._heat_norm())
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
            ("Main Clutch (in-piston)", "clutch_main", 0.0, 1.0, False),
            ("Transmission Bind", "clutch_trans", 0.0, 1.0, False),
            ("Coolant Flow", "coolant_flow", 0.0, 1.0, False),
            ("Gear (ring)", "gear_sel", 0, nrings - 1, True),
        ]

    def _bench_layout(self):
        px, py, pw = 12, 132, 278
        rects = []
        y = py + 40
        for _ in self._bench_specs():
            rects.append(pygame.Rect(px + 14, y, pw - 28, 8))
            y += 36
        return px, py, pw, y, rects

    def draw_bench_panel(self):
        specs = self._bench_specs()
        px, py, pw, yend, rects = self._bench_layout()
        panel_h = (yend - py) + 156
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
        k, t = self.kin, self.therm
        tcol = C_BAD if t["temp"] > THERM["warn_c"] else (C_WARN if t["temp"] > THERM["steam_c"] else C_GOOD)
        rows = [
            ("Piston", "%5.0f rpm" % k["piston"], C_RING_HOT),
            ("Core", "%5.0f  clutch slip %2.0f%%" % (k["core"], k["slip1"]), C_TEXT),
            ("Ratio (bind)", "%.2f:1  slip %2.0f%%" % (k["ratio"], k["slip2"]), C_TEXT),
            ("TransOut / Gen", "%5.0f / %5.0f" % (k["trans_out"], k["gen"]), C_GOOD),
            ("Gear / Fly", "%d of %d  |  %4.0f" % (k["gear"], k["ngear"], k["fly"]), C_TEXT),
            ("Block Temp", "%3.0f C" % t["temp"], tcol),
            ("Heat -> Cool", "%2.0f -> %2.0f kW" % (t["heat_kw"], t["cool_kw"]), C_TEXT_DIM),
            ("Elec  main+ORC", "%2.0f + %2.0f kW" % (t["main_kw"], t["orc_kw"]), C_GOOD),
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

    def draw_config_panel(self):
        n = self.renderer.n_pistons
        speed = "PAUSED" if self.paused else "%.2fx" % self.speeds[self.speed_idx]
        scol = C_WARN if self.paused else C_GOOD
        x, y, w = self.W - 312, self.H - 154, 300
        panel(self.screen, x, y, w, 134)
        self.screen.blit(self.fsmall.render("LIVE ENGINE CONTROLS", True, C_ACCENT), (x + 10, y + 8))
        rows = [
            ("PISTONS", "%d  (1-4)" % n, "[ ]", C_TEXT),
            ("COMBUSTIONS / REV", "%d  of 8" % self.combustions, ", .", C_RING_HOT),
            ("FIRING", self.fire_mode.title(), "M", C_ACCENT),
            ("VIEW SPEED", speed, "P  - =", scol),
        ]
        oy = y + 30
        for label, val, keys, col in rows:
            self.screen.blit(self.fsmall.render(label, True, C_TEXT_DIM), (x + 10, oy))
            self.screen.blit(self.fsmall.render(val, True, col), (x + 162, oy))
            self.screen.blit(self.fsmall.render(keys, True, C_TEXT_DIM), (x + 258, oy))
            oy += 24

    def draw_view_tabs(self):
        x, y = 12, 42
        for key, name, mode in (("1", "FULL", "full"), ("2", "EXPLODED", "exploded"),
                                ("3", "ASSEMBLY", "assembly")):
            cur = self.renderer.view == mode
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
            "Combustion: %d chambers firing  (%d /rev, %s)   Piston %.0f rpm   Output %.0f rpm" % (
                firing_now, self.combustions, self.fire_mode.title(),
                self.kin["piston"], self.kin["out"]), True, ccol), (14, self.H - 102))
        lines = [
            "POWER PATH:",
            "8-chamber ring -> 1st clutch (FULL=1:1 lock, no slip) -> Core shaft -> kinetic flywheel store",
            "-> gear-increase core (slip=ratio / lock=100% gear) -> 2nd clutch -> axial generator",
            "-> receival gear (~18 deg angle loss) -> output adapter (DIRECT or DIFFERENTIAL) -> wheels",
        ]
        panel(self.screen, 12, self.H - 86, 760, 74)
        for i, ln in enumerate(lines):
            self.screen.blit(self.fsmall.render(ln, True, C_ACCENT if i == 0 else C_TEXT),
                             (22, self.H - 80 + i * 17))

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
                             injection=self._injection, heat=self._heat_norm())

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
        cx = self.W // 2 + int(self.world.steer * 30)
        cy = self.H - 90
        body = pygame.Rect(cx - 90, cy - 36, 180, 70)
        pygame.draw.rect(self.screen, (40, 110, 180), body, border_radius=14)
        pygame.draw.rect(self.screen, (25, 70, 120), body, 2, border_radius=14)
        pygame.draw.rect(self.screen, (120, 180, 230),
                         (cx - 55, cy - 30, 110, 26), border_radius=8)
        for sx in (-66, 66):
            pygame.draw.circle(self.screen, (15, 15, 18), (cx + sx, cy + 36), 20)
            pygame.draw.circle(self.screen, (60, 60, 66), (cx + sx, cy + 36), 9)
        if self.brake > 0.1:
            for sx in (-86, 86):
                pygame.draw.circle(self.screen, (255, 50, 40), (cx + sx, cy + 6), 6)
        if self.world.gear < 0 and abs(self.world.v) > 0.2:   # reverse lights
            for sx in (-78, 78):
                pygame.draw.circle(self.screen, (240, 240, 230), (cx + sx, cy + 14), 5)

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

        panel(self.screen, self.W // 2 - 300, self.H - 40, 600, 30, alpha=170)
        eng = "ENGINE FIRING" if pt.engine_on else "ENGINE OFF (electric)"
        ecol = C_RING_HOT if pt.engine_on else C_GOOD
        grade_pct = math.tan(self.cur_grade) * 100
        regen = " REGEN>>" if pt.regen_active else ""
        orc = self.therm["orc_kw"]
        tcol = C_BAD if self.temp_c > THERM["warn_c"] else C_TEXT
        base = "%s%s   grade %+4.1f%%   %6.2f mi   " % (eng, regen, grade_pct, pt.miles)
        img = self.font.render(base, True, ecol)
        self.screen.blit(img, (self.W // 2 - 290, self.H - 36))
        self.screen.blit(self.font.render("%3.0fC  ORC %.0fkW" % (self.temp_c, orc),
                                          True, tcol), (self.W // 2 - 290 + img.get_width(), self.H - 36))

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
            lines = ["drag orbit | wheel zoom | right-drag pan | L labels | R reset view",
                     "hover a part = inspect | click = pin (or place next in assembly)",
                     "P pause | -/= speed | [ ] pistons | , . combustions/rev | M firing mode"]
            y0, wide = 70, 660
        else:
            lines = ["DRIVE  --  W/Up: throttle  S/Down: brake  A/D: steer",
                     "R: shift Drive<->Reverse (stopped) | C: cruise | E: engine burst",
                     "G: downhill | SPACE: hard brake | TAB: preview | I: full spec"]
            y0, wide = 204, 540   # under the speedo, clear of the bottom energy bars
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
    print("   TAB  switch PREVIEW <-> DRIVE      I  info panel     H  help")
    print("   PREVIEW VIEWS:  1=FULL  2=EXPLODED  3=ASSEMBLY (puzzle)")
    print("   PREVIEW: drag=orbit  wheel=zoom  right-drag=pan  L=labels  R=reset")
    print("            hover a part = read spec | click = pin / place next part")
    print("   PLAYBACK: P=pause  -/= view speed  [ ]=pistons  , .=combustions/rev")
    print("   DRIVE:   W/Up=throttle  S/Down=brake  A/D=steer  C=cruise")
    print("            E=engine burst  G=downhill  SPACE=hard brake")
    print("   ESC  quit")
    print("=" * 70)


def main():
    App().run()


if __name__ == "__main__":
    main()
