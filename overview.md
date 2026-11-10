# GmansRun V1.17 — Complete Technical Overview

`GmansRunV1.17.py` is a fully standalone Python simulator for the **HOHEV-Rotary
Gen 4** concept vehicle. It combines an inspectable 3D rotary-engine model, a
whole-car 3D model, a drivable road simulation, a full information/specification
overlay, and a real-life MPG-vs-speed estimate — all in one file (~7,700 lines),
with no external assets and only `numpy` + `pygame` as dependencies.

The project is a **conceptual engineering simulator**, not a CAD/CFD/FEM
validation package. It demonstrates the proposed mechanism, energy routing,
losses, harvest systems, scaling assumptions and optimizer-selected constants in
one runnable program. **Every constant in this document was re-verified against
the current source.**

---

## Table of Contents

1. [How To Run](#how-to-run)
2. [Repository Files](#repository-files)
3. [Program Architecture](#program-architecture)
4. [Dimension Tables (DIMS — everything to scale)](#dimension-tables-dims)
5. [Renderer + Performance Architecture](#renderer--performance-architecture)
6. [Core Concept](#core-concept)
7. [Wheel-Drive Clarification (EM rail-ring)](#wheel-drive-clarification-em-rail-ring)
8. [Runtime Modes](#runtime-modes)
9. [Engine Architecture](#engine-architecture)
10. [Thermal And Fluid Model](#thermal-and-fluid-model)
11. [Vehicle And Road-Load Model](#vehicle-and-road-load-model)
12. [Physics Equations](#physics-equations)
13. [Drive Physics And Energy Flow](#drive-physics-and-energy-flow)
14. [Energy Harvesting Stack (energy-honest)](#energy-harvesting-stack-energy-honest)
15. [Electrical Bus + ECU](#electrical-bus--ecu)
16. [Storage And Routing](#storage-and-routing)
17. [Predictive Control](#predictive-control)
18. [Per-Frame Pipeline](#per-frame-pipeline)
19. [Optimizers](#optimizers)
20. [MPG Chart Values](#mpg-chart-values)
21. [Controls](#controls)
22. [Engineering Interpretation](#engineering-interpretation)
23. [Caveats](#caveats)
24. [Glossary](#glossary)
25. [Maintenance Checklist](#maintenance-checklist)

---

## How To Run

```bash
python -m pip install pygame numpy     # only dependencies
python GmansRunV1.17.py                # run the simulator
```

Offline tools (import the sim headless — no window opens):

```bash
python optimize_harvest.py 300000      # sizing sweep (~300k configs)
python optimize_control.py             # operating point + SOC policy sweep
```

Headless checks:

```bash
python _smoke.py      # full UI/sim smoke test (SDL dummy driver) + per-mode fps
python _mpgtest.py    # storage-conservation cruise test
python -m py_compile GmansRunV1.17.py optimize_harvest.py optimize_control.py
```

## Repository Files

| File | Role |
|---|---|
| `GmansRunV1.17.py` | The entire simulator: constants, geometry builders, software renderer, drive physics, energy model, info pages, MPG chart, controls. **Single source of truth.** |
| `optimize_harvest.py` | Offline sizing optimizer (~30 knobs) for the harvest stack, aero/rolling, drivetrain, storage and the kinetic→electrical chain. |
| `optimize_control.py` | Offline operating-point + control-policy optimizer (firing RPM × combustions, then SOC policy over a demanding cycle). |
| `_smoke.py` | Headless smoke test: boots the app with SDL dummy driver, pumps frames in all three modes, exercises tab clicks, gear seating, clutch-slip sweep, prints per-mode fps. |
| `_mpgtest.py` | Conservation test: 600-s cruises at several speeds; asserts stored energy never rises without fuel. |
| `readmev2.py` / `readmev2.md` | **Sibling project** — GmanCoatV1.17, the single-spray coating simulator derived from this project's HYDRO skin model. Not part of the engine sim. |
| `referencecode/` | Earlier reference snapshots. |
| `Goal of GmansRunV1.17`, `Plan.md`, `plan2.md`, `MODEL_REFINEMENT.md` | Historical design notes. Some values are stale; executable constants win. |
| `image.png` | Ring-clutch transmission design reference. |
| `overview.md` / `README.md` | This document / the quick-start project readme. |

## Program Architecture

One file organized as: **constants → geometry builders → physics helpers →
state classes → pygame application loop.**

### Classes

| Class | Responsibility |
|---|---|
| `Mesh` | Vertices + faces + color + name + spin ratio + kinematic `group` + `pivot` + `tilt`. `world_verts()` applies spin → tilt → pivot. Flags: `hot`, `brake`, `reverse`, `chamber_index`, `selectable`, `two_sided` (auto-detected by `_mesh_cache` from edge sharing). |
| `Part` | Named bag of meshes + spec-card text + assembly `order` + `explode` offset + `popdir` + optional `fire_anchor`. |
| `EngineRenderer` | The software 3D engine: full / exploded / assembly views + section-cut toggle, orbit/zoom/pan, hover pick, click-to-pin, labels, fire/injection glow, studio floor (reflection + shadow + vignette). |
| `Powertrain` | Battery SOC, supercap kWh, flywheel kWh/RPM, fuel used, engine-on time, burst state, one-pass energy netting, storage routing. |
| `DriveWorld` | Procedural hilly route; signed acceleration from rail force, brakes, aero, rolling, gravity; regen on negative road power; grade + lookahead sampling. |
| `App` | Pygame setup, event handling, mode switching (PREVIEW / CAR / DRIVE), all panels and HUDs, MPG chart, info/help overlays, adaptive quality. |

### Key functions

- `build_engine_parts(n_pistons)` — the rotary-generator assembly.
- `build_car_parts()` — the whole vehicle and every visible efficiency part.
- `solve_engine_physics(k, combustions, temp_c, coolant_flow, hyd_cmd, firing, slip_load, n_pistons, boiler_c, fluid, fluid_health)` — one frame of combustion→shaft→generator→heat; returns a dict (`gross_kw`, `gen_kw`, `orc_kw`, `fuel_g_s`, `heat_in_boiler`, `recovery_eff`, `exh_stage2_kw`, `block_heat_kw`, …).
- `solve_electrical()` — HV bus current, voltage sag, component losses.
- `road_load_watts()` → (total, aero, rolling) W; `effective_cd(v_mph)`, `effective_crr()`; `estimate_mpg()` / `estimate_mpg_baseline()`.
- Harvest helpers: `ambient_harvest_w()`, `solar_boiler_kw()`, `suspension_bump_w()`, `teg_harvest_kw()`, `steering_regen_kw()`, `drag_capture_kw()`, `windmill_kw()`, `pendulum_harvest_w()` / `pendulum_deploy()`.
- Transmission helpers: `trans_ratios()`, `trans_ring_edges()`, `output_gear_mesh_radius[_f]()`.

## Dimension Tables (DIMS)

Every part is rendered **to scale** from `DIMS` (millimetres).

### Rotary combustion ring

| Key | Value | Meaning |
|---|---:|---|
| `ring_outer_d_mm` | 760 | ring outer diameter |
| `ring_inner_d_mm` | 520 | hollow bore (holds the transmission) |
| `ring_thickness_mm` | 165 | axial thickness (deliberately thin) |
| `chambers` | 8 | combustion paddles, 45° apart |
| `chamber_depth_mm` / `chamber_width_mm` | 70 / 120 | paddle radial depth / tangential width |

### Core driveline

| Key | Value | Meaning |
|---|---:|---|
| `flywheel_d_mm` / `flywheel_thick_mm` / `flywheel_mass_kg` | 460 / 90 / 58 | free-floating tungsten flywheel in the ring's hollow centre |
| `clutch1_d_mm` / `clutch1_thick_mm` | 480 / 22 | multi-plate wet clutch: ring → shaft |
| `shaft_d_mm` / `shaft_len_mm` | 70 / 360 | central shaft ("the Core") |
| `trans_rings` | 6 | concentric slip-ring "gears"; ratchet shifter slides up only |
| `trans_outer_d_mm` / `trans_inner_d_mm` / `trans_thick_mm` | 500 / 90 / 60 | transmission annulus |
| `gen_d_mm` / `gen_thick_mm` | 300 / 110 | axial-flux generator |
| `out_gear_d_mm` / `out_gear_thick_mm` / `out_angle_deg` | 34 / 34 / 14 | "receival" pinion at the unavoidable 14° angle-loss pivot |

### Boost + heat recovery

| Key | Value | Meaning |
|---|---:|---|
| `turbo_turbine_d_mm` / `turbo_comp_d_mm` / `turbo_len_mm` / `turbo_runner_d_mm` | 105 / 94 / 62 / 28 | exhaust turbocharger |
| `supercharger_d_mm` / `supercharger_len_mm` / `supercharger_pulley_d_mm` | 92 / 78 / 72 | slow automatic supercharger |
| `coolring_gap_mm` / `coolring_thick_mm` | 26 / 20 | cooling-ring standoff / jacket wall |
| `boiler_d_mm` / `boiler_len_mm` | 96 / 120 | compact boiler drum |
| `steam_chamber_d_mm` / `steam_stage_d_mm` | 44 / 54 | dual valve chambers / expander stages |
| `steam_gen_d_mm` / `steam_gen_thick_mm` | 86 / 58 | steam-driven generator |
| `relief_d_mm` | 34 | pressure-relief / vent turbine |
| `shield_span_mm` / `shield_vanes` | 150 / 7 | geometric heat shield + reflector vanes |
| `recov2_turbine_d_mm` / `recov2_gen_d_mm` | 78 / 64 | second-stage exhaust recovery |

### Vehicle + accessories

| Key | Value | Meaning |
|---|---:|---|
| `wheel_d_mm` / `wheel_gen_d_mm` / `wheel_width_mm` | 560 / 300 / 150 | engine-view wheel + in-hub regen disc |
| `seats` / `pedal_crank_d_mm` / `pedal_gen_d_mm` | 4 / 92 / 70 | pop-out pedal generators (one per seat) |
| `solar_roof_len_mm` / `solar_roof_width_mm` | 2200 / 1300 | PV roof panel |
| `windshield_lens_d_mm` / `lens_receiver_d_mm` | 520 / 72 | Fresnel lens aperture / boiler receiver |
| `flow_duct_width_mm` / `flow_duct_height_mm` / `flow_duct_len_mm` | 620 / 95 / 2850 | through-body wake duct |
| `CAR` dict | 3820 L × 1600 W × 1150 H, wb 2500, track 1350, wheel Ø 620×205, clearance 120, frontal fill 0.86 | whole-vehicle envelope |

## Renderer + Performance Architecture

`EngineRenderer.render()` is a numpy-vectorized painter's-algorithm software
renderer, profiled and restructured for frame rate. Current pipeline:

1. **Per-mesh transform cache** (`m._ck`): world+camera vertex arrays reused
   while (angle, explode offset, az/el/dist, pan, selector radius, viewport
   size) are unchanged. `spin == 0` meshes ignore group angle → cache across
   frames even while the engine animates.
2. **Per-mesh shading cache** (`m._pk → m._pp`): unchanged (transform, colour,
   outline, fog, section, rect) ⇒ reuse the whole polygon contribution — zero
   numpy work. CAR VIEW (mostly static meshes) ~tripled in frame rate.
3. **Vectorized culling**: near-plane clip, screen bounds, ≥3 px² sub-pixel
   cull, backface culling — all in numpy over `face_pad`; Python only touches
   faces that draw.
4. **One-shot point lists**: all surviving faces' point lists come from a
   single `tolist()` per mesh (padded vertices are harmless repeats).
5. **Vectorized shading + depth fog**: warm key + cool fill + rim + Blinn
   specular; distance fog blended into colours before the draw loop.
6. **Draw-loop LOD**: <60 px² fill-only; `aapolygon` AA edges above; dark facet
   outlines ≥140 px²; panel-gap tint ≥320 px² (skipped when an outline draws).
7. **Quality auto-scale**: AA only while fps ≥ 45; below 30 fps the model
   renders at 0.66× (0.5× under 18) and smoothscales up — **still re-rendering
   every frame** so motion stays fluid rather than freezing.
8. **Studio floor**: transparent work-surface mirror reflection + cached radial
   shadow sprite + cached vignette.
9. **Drive amortization**: chase car and live-engine inset render every other
   frame into cached surfaces.

Measured headless (dummy SDL, max-quality path): PREVIEW ~4 → ~7 fps,
CAR VIEW ~8 → ~22 fps; the degraded path is proportionally faster.

## Core Concept

A series-hybrid electric vehicle:

1. Fuel powers a circular **rotary combustion generator** (range extender only).
2. The generator makes electricity.
3. Electricity routes into supercapacitors, the structural battery, and a
   tungsten flywheel buffer.
4. Wheels are driven electrically by direct **EM rail-ring** wheel drives.
5. The engine **never** mechanically drives the wheels.

The design goal is **maximum engine-off driving**: the engine fires only in
short bursts at its single optimized operating point, when storage + live
harvesting can't cover projected demand.

## Wheel-Drive Clarification (EM rail-ring)

Not an in-wheel motor, not a separate electric engine. Per wheel:

- A stationary **96-pole hybrid superconducting + permanent-magnet rail
  stator** sits inside the rim across a **0.15 mm** air gap (outer-rotor —
  the rim IS the rotor, hard-bonded, no wheel clutch).
- Rail current 0–2500 A is ECU-slewed ≤ 6000 A/s → gearless, scalable torque.
- Same rail back-drives as a generator: **98% regen** braking/downhill.
- **Passive permanent-magnet bearings**: Halbach lift + diamagnetic
  (pyrolytic-graphite) pinning + low-drag ceramic touchdown race + eddy-current
  damper — Earnshaw-stabilized with **no powered coils**, cutting 45% of the
  rolling/bearing drag.
- Aggregate `VEH["drivetrain_eff"] = 0.994` = SiC inverter 0.997 × rail motor
  0.997. Max rail power 320 kW across 4 wheels; friction brakes to 9 kN.
- Standalone rail unit reference: 1.0 m disc, 950 kg, 350 kNm continuous /
  500 kNm capacitor-boosted burst, 25 krpm inner kinetic disc.
- All four rail-ring wheels are exploded/labelled in CAR VIEW (rim, rotor disc,
  magnet band, rail stator, windings, poles, gap callouts, bearing hub).

`RAILMOTOR` remains only as a back-compat alias of `RAIL_DRIVE`.

## Runtime Modes

`TAB` cycles **ENGINE PREVIEW → CAR VIEW → DRIVE**.

### Engine Preview

- Full / exploded / assembly / section-cut views; labels on `L`; hover inspect
  + click-to-pin spec cards; blueprint mode.
- Live mechanics: combustion glow at the fixed top injector, chamber fire
  order, wet-clutch engagement, ring-clutch bind sweep, selector travel,
  flywheel spin-up, heat reddening, power-path overlay.
- `[`/`]` rotary pistons 1–10 · `,`/`.` combustions/rev 1–8 · `F` boiler fluid ·
  `P` pause · `-`/`=` playback speed.
- **ENGINE TEST BENCH** sliders: engine power, hydraulic pressure, main clutch
  grip, ring bind, coolant flow, gear/ring select. Readouts: piston/core/gen
  RPM, slip, bind, gear, flywheel RPM, combustions/min, pressure ratio, AFR,
  chamber volume, clamp force, boost, friction, block/boiler temps, fluid
  pressure + health, shield heat flow, steam/exhaust recovery, electric
  output, ambient harvest, pedal assist.

### Car View

The whole vehicle to real-life scale, hero ¾ camera. Parts inventory: aero body
+ canopy, fenders, LED headlights, full-width rear light bar, structural
battery floor, supercap bank, 4 labelled rail-ring wheels (all internals),
solar roof + body film, windshield concentrator + sun boiler, flow-through
duct, drag-capture grooves, hydrophobic skin sample, windmill fin, regen
dampers, inertial pendulum rim, pedal generator, range-extender package.
Same view/section/assembly controls as preview.

### Drive Simulation

Procedural hilly route: throttle/brake/steer/cruise/burst/downhill/pedals.
Chase-camera render of the same whole-car model at reduced detail (0.7);
wheels spin at true road speed; brake/reverse lights are real meshes. HUD:
mph, live MPG, fuel used, SOC, supercap, flywheel RPM/energy, engine-on time,
rail power, regen + harvest flows, efficiency stack with live morphed-Cd badge.
Spinning engine in an every-other-frame inset.

### Info + MPG Chart

`I` = full specification overlay (parts, systems, constants, MPG logic,
storage, caveats). `M` = steady-state MPG-vs-speed chart (real road-load
physics, distinct from the arcade drive economy).

## Engine Architecture

Circular rotary piston — a **hollow ring** with 8 concave combustion paddles
(45° spacing), not a Wankel. Fuel → ring spin → clutch → shaft → transmission →
pinion → generator:

- 8 chambers, fixed top injector, compression gate + chamber timing.
- Integral multi-plate wet clutch inside the piston bore.
- **Multilayer ring-clutch transmission**: 6 concentric rings coupled through
  pressurized annular gaps; engagement sweeps centre-out, the outermost gap
  slips while the gear stays seated (high-pressure-coupling disengage); the
  ratchet shifter slides up only and seats the receival pinion.
- Central free-floating tungsten flywheel: 460 mm × 90 mm, 58 kg, 20 krpm cap.
- Output pinion at 14° angled pivot + 300 mm axial-flux generator.
- Cooling ring + geometric heat shield → 96 mm compact boiler; PCM bank.
- Turbocharger + slow automatic supercharger + 78 mm 2nd-stage turbine.
- Hydraulic actuation, 85 bar max; wet-clutch μ = 0.080.

**Optimized burst point (verified): 400 rpm × 8 combustions/rev** → 8.81 kW gen
+ 2.71 kW bounded same-second heat, 52.9% net fuel→electricity at the point.

**Speed ceiling**: `MAX_PISTON_RPM = 1200` — a 760 mm combustion ring tops out
at ~48 m/s rim speed (already aggressive vs. Wankel apex-seal speeds), so the
displayed kinematics stay mechanically plausible: the ring runs at hundreds of
rpm, not thousands, while the 34 mm receival pinion steps up ~3× (≈3,600 rpm at
the generator at full power). Other firing constants: `FIRING_IDLE_RPM = 320`,
`FIRING_GEN_RPM = 400`, `DRIVE_COMBUSTIONS = 8`, `FLYWHEEL_MAX_RPM = 20000`,
`MAX_PISTON_RPM = 1200` (was 4500 — corrected; the earlier ceiling implied a
~180 m/s rim speed).

## Thermal And Fluid Model

Closed-loop steam/ORC heat recovery — keep heat, milk pressure:

1. Combustion → shaft work + waste heat.
2. Exhaust turbo stage + second-stage turbine (28% of leftover exhaust).
3. Cooling ring + heat shield trap 95% of block heat → boiler (78% of the
   remaining exhaust also routed there).
4. Boiler vaporizes the working fluid; **dual-chamber 4-stage compound
   expander** (+12%/stage, ×1.18 alternating duty) milks pressure internally.
5. **PCM bank** (120 kJ/C) stores excess heat, releasing it for minutes after
   shutdown → electricity with the burner OFF.
6. Conditioning is a last-resort backstop at overheat (125/155/240 °C stages).

| Fluid | Boil °C | Full °C | Max bar | Freeze °C | Heat OK °C | Service kWh | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| **AMMONIA** (default) | 26 | 92 | 26 | −78 | 230 | 4200 | highest pressure + endurance |
| R245FA | 30 | 96 | 21 | −103 | 175 | 2600 | ORC refrigerant |
| METHANOL | 55 | 135 | 15 | −98 | 195 | 2000 | simple antifreeze |
| WATER | 82 | 165 | 14 | 0 | 240 | 1500 | comparison only — freezes! |

Recovery is bounded by same-second heat above the fluid's boiling onset and the
0.62 Carnot-bounded cap — no energy from ambient heat alone.

## Vehicle And Road-Load Model

Minimize road load first — the cheapest watt is the one never needed:

- `Cd = 0.08` base → effective ≈ **0.041–0.061** after morphing aero, hydro
  skin and flow duct (computed per speed; see equations below).
- `Crr = 0.0028` → effective ≈ **0.00154** after the 45% PM-bearing cut.
- Frontal area 1.58 m² (1600 × 1150 mm × 0.86 fill); curb 765 kg + 80 kg/occupant.
- 40 kWh structural battery inside the carbon monocoque.

Road power = aero (v³) + rolling (v¹) + grade; regen when negative. The aero
stack exists because aero dominates at speed: teardrop + boat-tail, covered
wheels, sealed underbody, morphing surfaces, plastron slip skin, wake duct.

## Physics Equations

**Road load** (`road_load_watts`, steady flat):

```
v        = mph × 0.44704
F_aero   = ½ · ρ(1.225) · Cd_eff(v) · A(1.58) · v²          [v² force → v³ power]
F_roll   = Crr_eff · m(765+80·n_pass) · g(9.81)
P_wheel  = (F_aero + F_roll) · v                            [watts]
```

**Effective Cd** (`effective_cd`, all factors multiply the base 0.08):

```
Cd_eff = 0.08 × morph(v) × hydro(RH) × duct(v) × pendulum_penalty(v)
morph: 1.0 → 0.74 ramped between 22 and 58 mph
hydro: 1 − (0.075 + 0.110 · HUMIDITY=0.6 → ~0.141 total)
duct:  1 − (0.15 + 0.10·min(v/60,1)) · 0.78
pend:  +0.012 Cd while the skirt is deployed (≤38→60 mph ramp)
```

**Effective Crr** (`effective_crr`): `0.0028 × (1 − 0.45) = 0.00154`.

**Combustion** (`solve_engine_physics`, first order): ideal-gas charge mass
from trapped pocket volume × pressure ratio × `volumetric_eff` → fuel mass at
AFR 18.5 → indicated work at `indicated_eff` 0.44 of LHV → mechanical 0.955 →
generator 0.975 → coupling 0.988; hydraulic clamp force sets wet-clutch torque
capacity; slip beyond capacity makes heat instead of torque; overheat derate
above 125 °C.

**Fuel economy** (`estimate_mpg`):

```
deficit      = max(0, P_wheel − P_free_external(solar etc.))
kWh_per_mile = deficit/1000 / v / drivetrain_eff + hotel/parasitics
MPG          = (33.6 kWh/gal × 0.60 fuel→wheel) / kWh_per_mile
             → solar-sustained (capped) when deficit = 0
```

**Storage honesty**: charge applies the round-trip factor on the way IN:
`E_stored += P_in·dt·rt`; discharge draws `P_out·dt` — net charge+discharge
always ≤ input. **One-pass netting**: same-frame harvest offsets traction
before touching storage, bypassing round-trip loss entirely.

## Drive Physics And Energy Flow

`DriveWorld.update()`: signed acceleration from rail force (≤320 kW, reverses
in R), friction brakes (≤9 kN), aero, rolling, grade gravity; negative road
power → rail regen. `lookahead_grade()` samples the route 260 m ahead.

`Powertrain.update()`: engine on/off from SOC thresholds + predictive target +
thermal state; tracks fuel, miles, engine seconds and every power flow
(`flow` dict: trac, regen, gen, steam, solar, susp, pedal, drag, wind, pend,
teg, steer…). Surplus routes supercap → battery → flywheel; deficit draws the
cheapest-loss source first.

## Energy Harvesting Stack (energy-honest)

Strictly conservative — three honest input classes:

1. **Fuel** — the only chemical input.
2. **External ambient** — genuinely enters from outside:
   - Solar film 210 W + solar roof 440 W = **650 W** nameplate (SUN-scaled);
   - Windshield Fresnel concentrator ≈ **0.18 kW thermal** — aperture-bounded
     (π·(0.52/2)² m² × 1 kW/m² × 0.85 optics): a lens concentrates flux
     density, not total power;
   - Parked windmill fin: ≤95 W, deploys ≤2 mph (WIND=0.55 ambient);
   - Passenger pedals: 110 W/seat × 4 seats;
   - Combustion waste heat via steam/ORC + TEG (heat the fuel already made).
3. **Recover-only** — reclaim a bounded fraction of energy ALREADY dissipated,
   → 0 in smooth steady cruise:
   - suspension dampers: 130 W baseline + ≤80 W bump-peak, bump dissipation only;
   - TENG 95 / tyre 55 / piezo 42 W nameplates, runtime-capped to **45% of the
     live rolling loss** (`HARVEST_ROLL_FRAC`) → 0 parked;
   - drag-air capture: grooves 0.28 × turbines 0.22 ≈ **6.2% of aero drag power**;
   - inertial pendulum rim: 28×0.55 kg bobs, 90 W peak; decel/cornering/pitch
     only — accelerating does NOT harvest (it would steal traction); deploys
     ≤38 mph, retracts ≥60 mph (+0.012 Cd when out);
   - steering regen 120 W; rail braking/downhill regen 98%.

Verified (`_mpgtest.py`): engine off + no sun ⇒ a steady cruise NEVER increases
stored energy; harvesters only slow the drain.

## Electrical Bus + ECU

Explicit HV model (`BUS` / `ECU` / `solve_electrical()`):

- 800 V nominal pack (~50 Ah for 40 kWh); `I = P/V`; 450 A contactor limit.
- SiC inverter 0.997 × rail motor 0.997 = **0.994** aggregate drivetrain eff.
- Superconducting HV bus (boiler cryo tap) → ~zero I²R cable loss.
- **ECU**: deterministic 200 Hz loop (5 ms), thresholds from `ENGINE_*_SOC` +
  `PREDICT_*`, rail-current slew ≤ 6000 A/s.
- Live pack V, bus A, losses and loop rate shown in the DRIVE panel.

## Storage And Routing

| Store | Size | Round-trip | Role |
|---|---:|---:|---|
| Supercapacitor | 0.40 kWh | 0.998 | fast pulses first (sized for power, not bulk — oversizing just adds ~125 kg/kWh) |
| Structural battery | 40 kWh | 0.992 | bulk storage, SOC window 15–52% |
| Tungsten flywheel | mech. buffer | 0.970 | mechanical overflow |

- Engine fires at **22% SOC**, stops at **42%** OR at **118 °C** block (steam
  keeps harvesting engine-off); refires below the **13%** critical floor even
  when hot. Window chosen to always keep regen headroom — never chase 100%.

## Predictive Control

`PREDICT_LOOKAHEAD_M = 260 m`, swing ±10 pts, saturating at 5% grade:

- **Climb ahead** → raise the engine-off SOC target (pre-charge; climb on
  stored energy so the engine stays off on the hill).
- **Descent ahead** → lower it (empty buffers so the whole downhill regenerates
  instead of heating brakes).
- Net: fewer burns, more captured free energy → higher MPG.

## Per-Frame Pipeline

```
run loop (60 Hz cap):
  clock.tick → dt (≤50 ms)
  handle_events        input, mode tabs, sliders, keys
  update               fps smooth → AA quality flag → renderer tick (pop/explode)
                       DriveWorld (drive mode) → Powertrain + engine physics
                       thermal + kinematics (gears, selector, gap fills)
  draw                 topbar → mode panel:
                         PREVIEW/CAR: _model_cached → EngineRenderer.render
                           (transform/shading caches → vectorized cull+shade
                            → painter sort → LOD draw → floor fx → labels)
                         DRIVE: road scene + chase car + engine inset + HUD
                       panels, chips, overlays → flip
```

## Optimizers

Both import the sim headless (SDL dummy) and reuse the real constants/physics.

### `optimize_harvest.py` — sizing

~30 knobs over solar W, Cd, frontal area, Crr, morphing aero, hydro skin,
bearing cut, duct, suspension, TENG/tyre/piezo, pendulum, drag capture, TEG,
drivetrain/regen/recovery/mechanical/generator/gear efficiencies, supercap +
battery sizing and round-trips. Fitness = fuel/mile over a weighted cycle
(15 mph×14%, 28×20%, 42×26%, 58×24%, 72×16%); every knob pays mass and (for
external parts) drag → real interior optima. Random search + coordinate descent.

- **Latest verification re-run**: baseline = optimum, +0.0% cycle MPG — the
  baked values ARE the optimum (~48 kg harvester mass at optimum, of 765 kg).
- Landmark earlier run: 301,568 configs, **+155.4%** cycle MPG vs old baseline,
  ~51.7 kg added.
- Fuel→electricity chain at the optimum: 44.0% indicated → 42.0% mech → 41.0%
  gen → 40.5% coupling, +7.5 pts steam recovery ⇒ **48.0% fuel→electricity**,
  × drivetrain ⇒ ~46.5% fuel→wheel in the optimizer's own bounds (the sim's
  integrated `FUEL_TO_WHEEL_EFF` uses 0.60 including ammonia bottoming).

### `optimize_control.py` — operating point + policy

- **Part A**: 3,288 firing points through the real `solve_engine_physics()`.
  Current = optimum: **400 rpm × 8 combustions/rev** → 8.81 kW gen + 2.71 kW
  bounded heat, 52.9% net. (Stored boiler heat is correctly NOT counted as
  same-second efficiency.)
- **Part B**: 85 SOC policies over a demanding scenario (85 mph, hills, sun
  0.05, low initial SOC). Optimum **on 0.22 / off 0.42 / smax 0.52 → ~770 MPG,
  engine-on 17.0%** (previously on=0.20 → 740 MPG/17.7%; the +4.1% value is
  now applied in the sim).

## MPG Chart Values

`M` chart — `estimate_mpg()`, steady flat road, favourable sun (SUN=1.0).
Solar-sustained points display capped `>MPG`, never "infinite":

| Speed | Estimate | Basis |
|---:|---:|---|
| 5 mph | >2000 capped | solar-sustained: 26 W road vs ~660 W free |
| 10 mph | >2000 capped | solar-sustained: 57 W vs ~673 W |
| 25 mph | >2000 capped | solar-sustained: 207 W vs ~708 W |
| 50 mph | >2000 capped | solar-sustained in bright sun: 746 W vs ~766 W |
| 80 mph | ~1250 MPG | aero-dominated; ~1.8× the ~694 MPG baseline |

Demanding drivable case (control optimizer): **~770 MPG @ 85 mph with hills +
low sun**, engine on ~17% of the time. Model outputs under stated assumptions —
not certification figures.

## Controls

| Context | Input | Action |
|---|---|---|
| Global | `TAB` | ENGINE PREVIEW → CAR VIEW → DRIVE |
| | `M` / `I` / `H` | MPG chart / info overlay / help |
| | `ESC` | quit |
| Camera | drag / wheel / right·middle drag / `Shift` | orbit / zoom-at-cursor / pan / fine |
| Views | `1` `2` `3` / `4` or `X` / `L` / `R` | full · exploded · assembly / section-cut / labels / reset |
| | hover / click | inspect part / pin spec card (assembly: place next part) |
| Playback | `P` `-` `=` `[` `]` `,` `.` `F` | pause · speed · pistons 1–10 · combustions 1–8 · fluid |
| Assembly | `N`/`Space`/`→` · `B`/`←`/`Bksp` · `F` · `0` | next · back · all · clear |
| Drive | `W`/`↑` `S`/`↓` `A`/`D` `Space` | throttle · brake · steer · hard brake |
| | `R` `C` `E` `G` `K` `F` | reverse (stopped) · cruise · burst · downhill · pedals · fluid |

## Engineering Interpretation

1. Minimize road load first (aero, rolling, bearings, duct, skin).
2. Drive the rims directly with rail stators — no gearbox/clutch loss.
3. Same rails regenerate at high rate (98%).
4. Fast pulses → supercaps; bulk → battery; overflow → flywheel.
5. Keep SOC low enough to always accept future regen.
6. Fire the rotary only at its optimized burst point, rarely.
7. Recover heat during AND after bursts (PCM stretches engine-off time).
8. Size every harvester where benefit beats mass/drag — synergy, not size.

## Caveats

- First-order concept simulator — not proof of buildability; no substitute for
  CAD, FEM, CFD, battery-safety validation, thermal sim or road testing.
- MPG outputs depend heavily on assumptions (Cd, area, Crr, mass, harvest W,
  weather, route, driving style).
- "Solar-sustained" = external solar covers road load at that instant; the
  model is energy-conservative throughout.
- Optimizer results are sweep results inside defined bounds; different
  bounds/route-mix/caps shift the optimum.
- The pygame software renderer is for explanation/inspection, not CAD export.

## Glossary

- **HOHEV** — Hybrid-Optimized High-Efficiency Vehicle (concept name).
- **Ring-clutch transmission** — concentric annular rings coupled through
  pressurized fluid gaps instead of toothed gears; slips centre-out.
- **Plastron** — trapped air layer on a super-hydrophobic surface acting as a
  slip boundary for airflow.
- **PCM** — phase-change material; stores heat as latent melt energy.
- **ORC** — organic Rankine cycle; low-boiling working fluid (ammonia here)
  instead of water steam.
- **TEG** — thermoelectric generator (Seebeck effect).
- **TENG** — triboelectric nanogenerator film.
- **Rail-ring drive** — stationary EM rail stator + rim-as-rotor, gearless.
- **One-pass netting** — using same-instant harvest for traction before it ever
  touches storage (no round-trip loss).
- **SOC window** — deliberate partial-charge band preserving regen headroom.
- **Solar-sustained** — instant where external solar ≥ road load: no fuel
  burns; displayed as a capped `>MPG`, not "infinite".

## Maintenance Checklist

1. Update constants in `GmansRunV1.17.py`.
2. `python optimize_harvest.py 300000` for sizing changes.
3. `python optimize_control.py` for firing/SOC/control changes.
4. `python _smoke.py` + `python _mpgtest.py` for regressions.
5. Check CAR VIEW parts + labels, and the `M` chart figures.
6. Update `INFO_SECTIONS` in the sim and this `overview.md`.
