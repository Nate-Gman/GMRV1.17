# GmansRun V1.17 Program Overview

`GmansRunV1.17.py` is a standalone Python simulator for the HOHEV-Rotary Gen 4
concept. It combines an inspectable 3D engine model, a whole-car 3D model, a
simple drivable road simulation, a full information/specification overlay, and a
real-life MPG-vs-speed estimate.

The project is intentionally a conceptual engineering simulator, not a finished
CAD/CFD/FEM validation package. It shows the proposed mechanism, energy routing,
losses, harvest systems, scaling assumptions, and optimizer-selected constants in
one runnable program.

## How To Run

Install the runtime dependencies:

```bash
python -m pip install pygame numpy
```

Run the simulator:

```bash
python GmansRunV1.17.py
```

Run the optimization helpers:

```bash
python optimize_harvest.py 300000
python optimize_control.py
```

Run a syntax check:

```bash
python -m py_compile GmansRunV1.17.py optimize_harvest.py optimize_control.py
```

## Repository Files

- `GmansRunV1.17.py`: main standalone pygame/numpy application. This contains
  the 3D part builders, renderer, drive physics, energy model, info pages, MPG
  chart, and controls.
- `optimize_harvest.py`: brute-force/randomized sizing optimizer for the energy
  harvesting stack, aero/rolling improvements, drivetrain efficiency, storage,
  and kinetic-to-electrical conversion chain.
- `optimize_control.py`: operating-point and control-policy optimizer. It sweeps
  generator RPM, combustions per revolution, SOC thresholds, and stress-route
  policy behavior.
- `Goal of GmansRunV1.17`: older project goal/specification note for the
  standalone digital twin.
- `Plan.md` and `plan2.md`: earlier design and implementation notes. Some values
  in these documents are historical. The current executable constants in
  `GmansRunV1.17.py` are the source of truth.
- `image.png`: design/reference image used during concept development.

## Program Architecture

The main simulator is one large Python file organized around constants, geometry
builders, physics helpers, state classes, and a pygame application loop.

Important data dictionaries:

- `DIMS`: engine, car, wheel, boiler, flywheel, pedal, solar, and duct dimensions.
- `PHYS`: first-order combustion, generator, clutch, hydraulic, heat, and loss
  constants.
- `THERM`: heat-trapping, boiler, second-stage turbine, compound-expander, and
  phase-change material constants.
- `FLUIDS`: selectable boiler working fluids and their boil/freeze/pressure
  behavior.
- `VEH`: vehicle mass, Cd, frontal area, rolling resistance, rail-drive power,
  and regen constants.
- `RAIL_DRIVE`: direct rim/rail drive details.
- `AMBIENT_HARVEST_W`, `AERO`, `HYDRO`, `WHEEL_BEARING`, `DRAG_CAPTURE`,
  `WINDMILL`, `PENDULUM`, `DUCT`, `TEG`, `ELEC`, and `STORAGE`: the main
  efficiency-stack configuration.

Important classes:

- `Mesh`: stores vertices, faces, color, name labels, spin behavior, group
  tags, pivots, and transforms for the custom 3D software renderer.
- `Part`: groups one or more meshes with a human-readable part name, spec-card
  text, assembly order, exploded-view offset, and color.
- `EngineRenderer`: renders engine or car parts in full, exploded, assembly, and
  section-cut modes. It handles camera orbit, pan, zoom, labels, hover picking,
  pinning, assembly placement, and painter-style face sorting.
- `Powertrain`: tracks battery SOC, supercap energy, flywheel energy, fuel used,
  engine-on time, engine burst state, storage routing, and live energy flows.
- `DriveWorld`: simulates vehicle motion on a procedural hilly route with aero,
  rolling, gravity, rail-drive force, braking, cruise control, regen, and
  predictive grade lookahead.
- `App`: owns pygame setup, event handling, mode switching, control panels,
  thermal state, drawing, HUDs, MPG chart, info overlay, and the main loop.

Important builders and helpers:

- `build_engine_parts()`: constructs the inspectable rotary-generator assembly.
- `build_car_parts()`: constructs the whole-car assembly and all visible vehicle
  efficiency components.
- `solve_engine_physics()`: calculates one frame of combustion/generator/thermal
  behavior from the current control state.
- `road_load_watts()`, `effective_cd()`, `effective_crr()`, and `estimate_mpg()`:
  steady-road physics and MPG helpers.
- `ambient_harvest_w()`, `suspension_bump_w()`, `teg_harvest_kw()`,
  `drag_capture_kw()`, `windmill_kw()`, `pendulum_harvest_w()`, and
  `steering_regen_kw()`: individual recovery-source helpers.

The optimizers import `GmansRunV1.17.py` directly with SDL set to dummy video and
audio drivers, so they reuse the simulator constants and physics without opening
the pygame window.

## Core Concept

The vehicle is modeled as a series-hybrid electric vehicle:

1. Fuel powers a circular rotary combustion generator.
2. The generator makes electricity.
3. Electricity is routed into supercapacitors, the structural battery, and a
   flywheel buffer.
4. The wheels are driven electrically by direct electromagnetic rail-ring wheel
   drives.
5. The engine does not mechanically drive the wheels.

The goal is maximum engine-off driving. The combustion engine runs only in short,
efficient bursts when storage and live harvesting cannot cover the projected
demand. The rest of the time the car tries to move from stored charge, solar,
regen, suspension harvesting, heat recovery, passenger pedal input, and other
small recovered sources.

## Important Wheel-Drive Clarification

The wheel system is not a conventional in-wheel motor and not a separate electric
engine.

Each wheel is modeled as an electromagnetic rail-ring direct drive:

- A stationary electromagnetic rail stator sits inside the rim across the
  precision air gap (an outer-rotor arrangement -- the rim itself is the rotor,
  not a self-contained bolt-in motor).
- The wheel rim and bonded rotor band act as the rotor and spin around it.
- The rail directly rotates the wheel itself.
- The rim, rotor disc, magnet band, rail stator, air gap, winding, poles, and
  passive magnetic bearing hub are now visible and labeled in CAR VIEW.
- The same rail-ring assembly back-drives during braking/downhill events to
  regenerate electricity.
- `RAIL_DRIVE` is the current code object. `RAILMOTOR` remains only as a
  compatibility alias for older helper code.

Current rail-drive constants include:

- 96-pole hybrid superconducting/permanent-magnet rail.
- 0.15 mm precision rail-to-rotor air gap.
- 2500 A maximum ECU-modulated rail current.
- 97.0% modeled rail drivetrain efficiency.
- 92% braking/downhill regen fraction.
- 320 kW total maximum rail-drive power in the drive simulation.

## Main Runtime Modes

Press `TAB` to cycle through the three primary modes.

### Engine Preview

The startup view is a navigable 3D model of the rotary generator. It supports:

- Full, exploded, and assembly views.
- Section-cut view.
- Labels.
- Hover inspection and click-to-pin spec cards.
- Live combustion glow, injector behavior, clutch/transmission movement, flywheel
  spin, fluids, heat systems, and power-path visualization.
- Piston count and combustion-count controls.

The left ENGINE TEST BENCH panel has draggable sliders for:

- Engine power.
- Hydraulic pressure.
- Main wet clutch grip.
- Transmission wet-clutch bind.
- Coolant flow.
- Selected gear/ring shift.

Its live readouts show piston/core/generator RPM, clutch slip, ring bind, gear,
flywheel RPM, combustions per minute, pressure ratio, air/fuel flow, chamber
volume, hydraulic clamp force, boost, friction, block/boiler temperature, boiler
fluid pressure, fluid health, shield-to-boiler heat flow, recovered steam/exhaust
power, electric output, ambient harvest, pedal assist, and thermal state.

### Car View

CAR VIEW shows the whole vehicle to real-life scale, framed by default from a
3/4-above hero camera. Every part is dimensioned to fit the body envelope: the
wheels are outer-rotor in-wheel drives whose EM rail stator, rotor magnet band,
windings and bearing hub all sit inside the tyre radius, so nothing protrudes
below the sealed underbody or past the body sides. In FULL view the vehicle reads
as one clean shape (labels appear on hover); EXPLODED/ASSEMBLY label every part.
It includes:

- Aerodynamic body and canopy.
- Body-coloured wheel-arch fenders.
- LED headlights and a full-width rear light bar (brake/reverse-lit while driving).
- Structural battery floor.
- Supercapacitor bank.
- Four labeled EM rail-ring direct-drive wheels.
- In-rim rotor/flywheel discs.
- Rotor magnet bands.
- Stationary EM rail stators.
- Copper rail windings.
- Pole markers.
- 0.15 mm air-gap callouts.
- Passive permanent-magnet bearing hubs.
- Standalone rail-drive unit/cross-section.
- Solar roof and upper-body solar film.
- Windshield solar concentrator.
- Sun boiler.
- Through-body flow duct.
- Drag-air capture skin/grooves.
- Hydrophobic/plastron skin sample.
- Windmill fin/parked airflow harvester.
- Regenerative suspension dampers.
- Inertial pendulum rim and retracting weights.
- Pedal-assist generator elements.
- Generator/range-extender package.

This mode is where the rim-drive mechanism should be inspected. Hover or click
the wheel/rail components to see the function/spec card.

### Drive Simulation

DRIVE mode is a simple drivable route with hills, throttle, braking, steering,
cruise, engine burst, downhill mode, and optional pedal assist. The car on the
road is the SAME 3D whole-vehicle model as CAR VIEW, rendered from a fixed
3rd-person chase camera (built at reduced mesh detail to protect the drive-loop
frame rate). Its wheels spin at the true road speed, the view banks and slides
with the steering, and brake/reverse lights plus the live aero badge overlay it.
It displays:

- Speed.
- Live MPG.
- Fuel used.
- Battery SOC.
- Supercapacitor charge.
- Flywheel RPM/energy.
- Engine-on time.
- Rail-drive power.
- Regen and harvest flows.
- Aero/rolling efficiency stack.

### Info Overlay

Press `I` to open the full informational specification. It explains the major
mechanical parts, energy systems, current optimized constants, MPG logic, storage
routing, and caveats.

### MPG Chart

Press `M` to open the real-life MPG-vs-speed estimate. This is separate from the
game-like drive loop. It uses steady flat road-load physics and shows why slow
driving can become fuel-free when live harvest covers the entire road load.

## Engine Architecture

The modeled generator is a circular rotary piston design, not a triangular Wankel.
The rotating ring is shaped like a hollow wheel with concave combustion chambers.

Major engine components:

- 8 combustion chambers around the circular piston.
- Fixed top injector.
- Compression gate and chamber timing.
- Integral multi-plate wet clutch inside the piston bore.
- Concentric transmission rings.
- Selectable inner-tooth ratios.
- Output pinion/generator path.
- Central tungsten flywheel.
- Cooling ring.
- Heat shield.
- Steam/boiler heat recovery.
- Phase-change heat bank around the boiler.
- Second-stage turbine/expander.
- Hydraulic clutch/actuation system.
- Turbo/supercharger visualization.
- Ambient harvest and auxiliary electrical systems.

The engine is generator-only in the current architecture. It is used as a
range-extender: fuel -> rotary kinetic work -> generator electricity -> storage
and rail-drive power.

Current optimized drive operating point:

- `FIRING_GEN_RPM = 400.0`
- `DRIVE_COMBUSTIONS = 8`
- Central tungsten flywheel: 460 mm diameter, 90 mm thick, 58 kg.
- Fuel-to-electricity target in optimizer output: about 48.0%.
- Fuel-to-wheel modeled constant: `FUEL_TO_WHEEL_EFF = 0.465`.

The "0 loss" idea is treated as an engineering target, not as the simulated
result. The current model explicitly includes losses through combustion,
mechanical transfer, generator conversion, coupling, storage, and rail drive.

## Thermal And Fluid Model

The heat-recovery model is a closed-loop steam/ORC-style system. It tries to keep
useful heat in the system instead of dumping it immediately to ambient air.

Modeled heat path:

1. Combustion produces shaft work plus waste heat.
2. A first exhaust/turbo stage recovers part of the exhaust stream.
3. A second exhaust turbine recovers part of the remaining exhaust energy.
4. The cooling ring and geometric heat shield direct block heat into the boiler.
5. A compact boiler heats the selected working fluid.
6. A dual-chamber compound expander reuses pressure through multiple stages.
7. The phase-change material bank stores excess heat and releases it after the
   engine shuts off.
8. Thermal conditioning only becomes a backstop when the block overheats.

Selectable working fluids:

- `AMMONIA`: default, low boiling point, high pressure, no-freeze behavior, long
  service life.
- `R245FA`: refrigerant-style ORC fluid.
- `METHANOL`: simple antifreeze-style fluid.
- `WATER`: allowed for comparison, but freezes at 0 C and needs higher heat to
  make useful pressure.

The solver does not allow heat recovery to create energy from ambient heat alone.
Useful work is calculated from heat above the ambient/boiling onset and bounded by
the fluid and expander efficiency assumptions.

## Vehicle And Road-Load Model

The car model is optimized around low road load first, because the cheapest watt
is the watt the vehicle never needs.

Current key values:

- Drag coefficient base: `Cd = 0.08`.
- Frontal area: `1.58 m2`.
- Rolling resistance coefficient: `Crr = 0.0028`.
- Passive magnetic wheel bearing rolling cut: 18%.
- Rail drivetrain efficiency: 97.0%.
- Curb mass: 765 kg.
- Structural battery: 40.0 kWh.

Road load is modeled from:

- Aerodynamic drag, which grows with velocity squared as force and velocity cubed
  as power.
- Rolling resistance, which scales mostly with weight and speed.
- Grade force on hills.
- Drivetrain losses.
- Regen when road power becomes negative through braking/downhill events.

At highway speed, aero dominates. That is why the model prioritizes:

- Narrow/low frontal area.
- Low Cd body.
- Active morphing aero.
- Hydrophobic/plastron slip skin.
- Through-body wake duct.
- Covered/optimized wheel regions.
- Low rolling resistance.

## Drive Physics And Energy Flow

In DRIVE mode, `DriveWorld.update()` calculates the signed vehicle acceleration
from:

- Direct rail-drive force at the wheels.
- Friction braking.
- Aerodynamic drag.
- Rolling resistance.
- Gravity on the procedural route grade.

The rail-drive force is power-limited by `VEH["max_rail_kw"]` and reverses when
the car is in reverse gear. Braking and downhill motion can produce negative road
power, which is routed as rail-regenerated electricity.

`Powertrain.update()` then decides whether the rotary generator should run. It
tracks:

- Fuel gallons.
- Miles.
- Engine seconds and total seconds.
- Engine on/off state.
- Battery SOC.
- Supercap energy.
- Flywheel energy.
- Engine, traction, regen, and flywheel power flows.

The app performs one-pass energy netting: generator, steam, solar, suspension,
pedal, regen, and other same-frame harvest power are first netted against
traction demand. Energy used immediately for traction bypasses storage, so it does
not pay a battery or supercap round-trip loss. Only true surplus is stored, and
only true deficit is pulled from storage.

## Energy Harvesting Stack (energy-honest)

The model is strictly energy-conservative -- no source creates energy from nothing.
There are only three honest classes of input:

1. **Fuel** -- burned in the rotary generator (the only chemical input).
2. **External ambient** -- energy that genuinely enters the vehicle from outside:
   solar (sun), parked wind, passenger pedal (human metabolic), and combustion
   waste heat (steam/ORC and TEG recover heat the fuel already produced).
3. **Recover-only** -- the motion "harvesters" do NOT add free power; each one can
   only RECLAIM a bounded fraction of energy the vehicle is ALREADY dissipating, so
   they fall to ~zero in a smooth, steady cruise:
   - Suspension dampers: reclaim BUMP dissipation only (-> 0 on a smooth road).
   - Triboelectric / tyre / piezo films: reclaim a capped fraction (`HARVEST_ROLL_FRAC`,
     12%) of the tyre/body hysteresis already lost to rolling resistance (-> 0 parked).
   - Drag-air capture: reclaims ~3% of the aero drag power already lost to the wake.
   - Inertial pendulum: reclaims body-rocking energy on DECELERATION, cornering
     scrub and pitch changes only -- accelerating does NOT harvest (it would steal
     traction). ~0 on a dead-straight steady cruise.
   - Rail-drive braking/downhill regen and steering regen recover otherwise-lost
     kinetic energy.

Verified: with no sun and the engine off, a steady flat cruise NEVER increases
stored energy -- the harvesters only slow the battery drain, never reverse it.

Current physically-grounded values (favorable daylight):

- Solar film: 210 W (~3 m^2 quantum-dot film x ~7% effective).
- Solar roof: 440 W (~2.0 m^2 PV x 22% x 1 kW/m^2 -> ~650 W total solar).
- Windshield sun-lens to boiler: ~0.18 kW THERMAL, bounded by the lens APERTURE
  (0.21 m^2 x 1 kW/m^2 x optics) -- a lens concentrates flux density, not total power.
- Suspension: 130 W nameplate, runtime = bump dissipation only.
- Triboelectric 62 / tyre 40 / piezo 28 W nameplate, runtime capped to 12% of the
  live rolling loss.
- Pendulum peak: 90 W, deceleration/cornering/pitch only.
- TEG scale: 0.0040 kW per degC, capped (recovers combustion waste heat).

## Electrical Bus + ECU

The high-voltage electrical system is modeled explicitly (not a black-box
efficiency), in `BUS` / `ECU` and `solve_electrical()`:

- 800 V nominal HV pack (`BUS["pack_v_nom"]`), ~50 Ah for the 40 kWh pack.
- Bus current is `I = P / V` from the demanded traction (or regen) power; a 450 A
  contactor/fuse limit is checked.
- The main copper cable has a real cross-section (50 mm^2) and length (4 m), so it
  drops `I^2 * R` as heat (`BUS_R_OHM = rho * L / A`).
- Inverter (98.5%) x motor (98.5%) = 0.970 -- i.e. these ARE the components the
  aggregate `VEH["drivetrain_eff"]` of 0.97 is made of, now shown explicitly with
  the cable drop on top.
- The ECU control logic (engine on/off, regen split, SOC window, predictive grade)
  runs at a fixed 200 Hz loop (`ECU["loop_hz"]`, 5 ms period) and slews the rail
  current at a bounded rate.

The live pack voltage, bus current, I^2R + inverter loss and ECU loop rate are shown
in the DRIVE efficiency-stack panel.

## Storage And Routing

The storage model is built around keeping regen headroom available.

Current values:

- Battery SOC minimum: 15%.
- Battery SOC maximum: 52%.
- Engine-off target SOC: 42%.
- Engine-on SOC threshold: 20%.
- Critical SOC safety floor: 13%.
- Engine-off heat threshold: 118 C block temperature.
- Supercapacitor buffer: 0.40 kWh.
- Supercap round-trip efficiency: 98.5%.
- Battery round-trip efficiency: 95.5%.
- Flywheel round-trip efficiency: 90.5%.

Routing philosophy:

- Fast pulses go to the supercapacitor first.
- Bulk energy goes to the battery only when there is room.
- Spillover can go into the flywheel buffer.
- The engine stops once the optimized SOC target is reached or when heat/steam
  recovery can carry more of the load.
- The system avoids filling the battery to 100%, because full storage would waste
  future downhill/braking opportunities.

## Predictive Control

The drive controller looks ahead along the procedural route before changing the
engine target.

Current predictive-control constants:

- Lookahead distance: 260 m.
- Maximum SOC target swing: +/- 10 percentage points.
- Grade reference for full swing: 5%.

Control behavior:

- If a climb is ahead, the controller raises the engine-off SOC target so the car
  can pre-charge and climb from stored energy.
- If a descent is ahead, the controller lowers the target to create empty storage
  headroom before downhill regen arrives.
- The engine fires when storage is low and stops when the SOC target is reached or
  when the block is hot enough for continued steam/TEG recovery.
- A critical low SOC can force the engine back on even if the block is hot.

## Optimizers

### `optimize_harvest.py`

This script searches over component sizes and performance constants, then scores
net drive-cycle MPG after mass, drag, and round-trip penalties. It tests the
synergy between:

- Solar sizing.
- Aero Cd and frontal area.
- Rolling resistance.
- Morphing aero onset/full-deploy behavior.
- Hydrophobic/plastron skin.
- Flow-through duct.
- Regenerative suspension.
- TENG/tire/piezo harvest.
- Pendulum peak output.
- Drag capture.
- TEG scale.
- Rail drivetrain efficiency.
- Regen fraction.
- Generator/mechanical/coupling efficiencies.
- Supercap/battery sizing and round-trip efficiency.

Latest documented large run:

- 301,568 configurations found the current expanded optimization package.
- Cycle MPG improved by about +155.4% versus the earlier baseline.
- Added optimization mass was about 51.7 kg.
- After applying the new constants, a 301,344-config re-run showed the current
  constants already matched the searched optimum.

### `optimize_control.py`

This script checks the engine operating point and SOC control policy.

It also corrected an earlier reporting problem: stored boiler heat is no longer
counted as same-second fuel efficiency. Heat recovery is bounded by actual
same-second heat flow plus direct exhaust-stage recovery.

Latest documented result:

- 3288 firing points swept.
- Current and optimum point matched at 400 rpm and 8 combustions/rev.
- Generator output: about 8.81 kW.
- Bounded heat recovery: about 2.71 kW.
- Net operating-point efficiency: about 52.9%.
- Stress/high-speed case: about 738 MPG with engine on about 17.8% of the time.

## MPG Chart Values

The `M` chart estimates steady, flat, favorable-condition fuel MPG at selected
speeds. When EXTERNAL solar (in good sun) covers the whole road load, no fuel
burns, so the point is **SOLAR-SUSTAINED** -- shown as a capped `>MPG` figure, NOT
"infinite MPG" (that framing read as magic). The tyre/tribo harvest cannot make a
point solar-sustained on its own -- it only reclaims rolling loss.

Approximate current chart figures (steady flat, favorable sun):

| Speed | Estimate | Note |
| ---: | ---: | ---: |
| 5 mph | >2000 (capped) | solar-sustained |
| 10 mph | >2000 (capped) | solar-sustained |
| 25 mph | >2000 (capped) | solar-sustained |
| 50 mph | ~2.1x baseline | solar covers it in bright sun |
| 80 mph | ~715 MPG | aero-dominated; ~1.4x baseline |

These chart values are not a promise of real-world certification MPG. They are a
model output for a specific steady flat scenario with the current internal
assumptions. In the drivable stress case, the control optimizer reports about
738 MPG at high speed/hills/low sun.

## Startup Output

When launched from a terminal, the program prints a startup banner summarizing:

- The three main modes.
- The current efficiency stack.
- Direct EM rail-ring wheel drive.
- The 300k+ sizing-optimization result.
- The kinetic-to-electrical chain.
- Storage round-trip routing.
- One-pass same-frame energy netting.
- Main controls.

## Controls

Global:

- `TAB`: cycle ENGINE PREVIEW -> CAR VIEW -> DRIVE.
- `M`: MPG chart.
- `I`: information/specification overlay.
- `H`: help overlay.
- `ESC`: quit.

Preview and Car View:

- Drag mouse: orbit.
- Mouse wheel: zoom.
- Right/middle drag: pan.
- `Shift`: fine camera movement.
- `1`: full view.
- `2`: exploded view.
- `3`: assembly view.
- `4` or `X`: section cut.
- `L`: labels.
- Hover part: inspect.
- Click part: pin/unpin or place next assembly part.
- `R`: reset camera/view.

Engine Preview only:

- `P`: pause animation.
- `-` / `=`: playback speed.
- `[` / `]`: piston count.
- `,` / `.`: combustions per revolution.
- `F`: boiler working fluid.

Assembly view:

- `N`, `Space`, or `Right`: place next part.
- `B`, `Left`, or `Backspace`: go back.
- `F`: place all.
- `0`: clear assembly.

Drive mode:

- `W` or `Up`: throttle.
- `S` or `Down`: brake.
- `A` / `D`: steer.
- `Space`: brake.
- `R`: drive/reverse when stopped.
- `C`: cruise.
- `E`: engine burst.
- `G`: downhill mode.
- `K`: passenger pedal assist.
- `F`: boiler working fluid.

## Current Engineering Interpretation

The best arrangement in the current model is:

1. Minimize road load first with body, aero, rolling, bearings, duct, and skin.
2. Drive the wheel rims directly with rail-ring stators to avoid gearbox/clutch
   losses at the wheel.
3. Use the same wheel rails for high-rate regen.
4. Capture fast energy in supercaps before sending bulk energy to the battery.
5. Keep battery SOC low enough to accept future regen.
6. Run the rotary generator only at its optimized efficient burst point.
7. Use heat recovery after and during bursts.
8. Stack small harvesters, but keep each one sized only where its benefit beats
   its mass/drag penalty.

This is why the optimizer favors synergy instead of simply making every harvester
as large as possible.

## Caveats And Correctness Notes

- This is a first-order simulator for concept exploration.
- It is not proof that the physical vehicle can be built to these outputs.
- It does not replace CAD, finite-element stress analysis, CFD, battery safety
  validation, thermal simulation, manufacturing review, or real dynamometer/road
  testing.
- The MPG estimates depend strongly on assumptions for Cd, frontal area, rolling
  resistance, mass, harvest watts, rail-drive efficiency, weather, route, and
  driving style.
- "Solar-sustained" (capped `>MPG`) in the chart means EXTERNAL solar covers the
  road load at that point so no fuel burns -- it is fuel-free at that instant, not
  unlimited or free energy. The model is energy-conservative: no source creates
  energy, and the motion harvesters only reclaim energy already being dissipated.
- Some older notes mention conventional in-wheel motors. The current model has
  been updated to direct EM rail-ring wheel drive.
- Some earlier values in `Plan.md`, `plan2.md`, and `Goal of GmansRunV1.17` are
  historical and may not match the current executable constants.
- The custom renderer is a pygame software renderer, so it is meant for visual
  explanation and inspection, not production CAD export.
- Optimizer results are stochastic/parameter-sweep results within the defined
  bounds. Changing bounds, weights, route mix, or physical caps can change the
  optimum.

## Maintenance Checklist

When changing the model:

1. Update constants in `GmansRunV1.17.py`.
2. Re-run `python optimize_harvest.py 300000` for system sizing changes.
3. Re-run `python optimize_control.py` for firing/SOC/control changes.
4. Re-run `python -m py_compile GmansRunV1.17.py optimize_harvest.py optimize_control.py`.
5. Check CAR VIEW to confirm all parts are visible and labeled.
6. Check the `M` chart and update any documented MPG values.
7. Update `INFO_SECTIONS` in the main program if user-facing specs changed.
8. Update this `overview.md` if architecture, constants, controls, or optimizer
   results changed.
