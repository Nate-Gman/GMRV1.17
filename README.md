# GmansRun V1.17 — HOHEV-Rotary Gen 4 Digital Twin

A **100% standalone** Python digital twin of a series-hybrid electric vehicle
concept: it builds, animates, and *drives* the entire HOHEV-Rotary powertrain
mechanically, to scale, in real time — in a single file with no external assets.

```bash
python -m pip install pygame numpy
python GmansRunV1.17.py
```

Requires Python 3.10+, pygame 2.x, numpy. Nothing else.

For the exhaustive technical reference — every constant, dimension table,
physics equation, optimizer result and caveat — see **[overview.md](overview.md)**.

---

## What You Get

| Mode | How to reach | What it is |
|---|---|---|
| **ENGINE PREVIEW** | loads first (`TAB` cycles) | Navigable 3D rotary engine: the 8-chamber combustion ring fires at a fixed top injector, the integral wet clutch engages toward 1:1, the **multilayer ring-clutch transmission** couples concentric rings through pressurized gaps (centre-out slip while gears stay seated), the free-floating tungsten flywheel winds up, and the axial-flux generator + angled "receival" pinion spin. Orbit / zoom / pan; full / exploded / assembly / section-cut views; hover spec cards; a live draggable **engine test bench**. |
| **CAR VIEW** | `TAB` | The whole vehicle to real-life scale (3820×1600×1150 mm). Every efficiency component is a labeled, dimensioned part — including all four **EM rail-ring** wheel drives shown down to their 0.15 mm air gaps, pole markers and bearing hubs. |
| **DRIVE** | `TAB` | Drive it over a procedurally hilly route: throttle/brake/steer/cruise, engine bursts, downhill regen, passenger pedal assist — with live battery SOC, supercap buffer, flywheel wind-up, engine-off time and a live MPG counter chasing the 1000+ MPG concept target. The same whole-car model runs under a chase camera; the spinning engine runs in an inset. |

Overlays: `I` full specification/info pages · `M` real-life MPG-vs-speed chart ·
`H` help.

## The Concept In One Paragraph

Fuel never touches the wheels mechanically. A circular **rotary combustion
ring** (8 concave chambers, one fixed injector) fires in short bursts only at
its optimizer-verified sweet spot (400 rpm × 8 combustions/rev, ~53% net
fuel→electricity at the point), spins a 58 kg tungsten flywheel through a wet
clutch and the ring-clutch transmission, and drives an axial-flux generator.
Electricity is netted against traction demand in **one pass** (used harvest
never pays a storage round-trip), then buffered supercap → battery → flywheel.
Each wheel rim **is** the motor rotor — a 96-pole superconducting/PM rail
stator inside the rim torques it gearlessly at ~99.4% and regenerates at 98%.
A waste-heat stack (turbo, 2nd-stage turbine, cooling ring, geometric heat
shield, closed-loop ammonia boiler, 4-stage dual-chamber compound expander,
PCM heat bank, TEG) keeps making electricity long after the burner stops.
Around that: solar roof+film (650 W), windshield concentrator, morphing aero,
hydrophobic/plastron slip skin, a through-body wake duct, passive
permanent-magnet bearings, drag-air capture, regenerative dampers, an
auto-deploying inertial pendulum rim, steering regen and passenger pedals —
every harvester sized by an optimizer so its benefit beats its mass/drag
penalty. The model is **energy-conservative**: nothing creates energy; motion
harvesters only reclaim what's already being dissipated.

## Numbers That Matter

| Metric | Value |
|---|---|
| Engine burst point | **400 rpm × 8 combustions/rev** (swept optimum, +0.0% headroom) |
| Ring speed ceiling | `MAX_PISTON_RPM` **1200** — ~48 m/s rim speed on the 760 mm ring; pinion ×3 → ~3,600 rpm gen |
| Point efficiency | 8.81 kW gen + 2.71 kW bounded heat, **52.9%** fuel→electricity |
| Integrated fuel→wheel | `FUEL_TO_WHEEL_EFF` **0.60** (indicated 0.44 × mech 0.955 × gen 0.975 × coupling 0.988 × rail 0.994 + ~13 pts heat bottoming) |
| Vehicle | 765 kg, Cd 0.08 → **~0.041–0.061 effective**, Crr 0.0028 → **~0.00154**, frontal 1.58 m² |
| Drivetrain | 320 kW rail-ring, 0.994 eff, 0.98 regen, 9 kN brakes |
| Storage | 40 kWh battery (SOC window 15–52%), 0.40 kWh supercap (RT 0.998 / 0.992 / 0.970) |
| Control | engine on @ **0.22** SOC, off @ **0.42** or 118 °C block, 260 m predictive lookahead |
| MPG chart | solar-sustained (capped) 5–50 mph, **~1250 MPG @ 80 mph**; ~770 MPG in the demanding 85-mph hills cycle |

## Controls

```
GLOBAL    TAB mode cycle   M mpg chart   I info   H help   ESC quit
CAMERA    drag orbit · wheel zoom-at-cursor · right/middle pan · Shift fine
VIEWS     1 full · 2 exploded · 3 assembly · 4/X section-cut · L labels · R reset
          hover inspect · click pin spec card / place assembly part
PLAYBACK  P pause · -/= speed · [ ] pistons 1-10 · , . combustions 1-8 · F fluid
ASSEMBLY  N/Space/→ next · B/←/Bksp back · F all · 0 clear
DRIVE     W/↑ throttle · S/↓ brake · A/D steer · Space hard brake
          R reverse (stopped) · C cruise · E engine burst · G downhill · K pedals
TEST BENCH  drag the six sliders: power, hydraulics, clutch grip, ring bind,
            coolant, gear select — live physics readouts beside them
```

## Repository Layout

```
Car Engine/
├── GmansRunV1.17.py      # THE simulator — single source of truth (~7.7k lines)
├── optimize_harvest.py   # offline sizing optimizer (imports the sim headless)
├── optimize_control.py   # offline operating-point + SOC policy optimizer
├── _smoke.py             # headless UI/sim smoke test + per-mode fps probe
├── _mpgtest.py           # energy-conservation cruise test
├── overview.md           # FULL technical reference (constants, dims, physics,
│                         #   optimizer results, renderer internals, glossary)
├── README.md             # this file
├── readmev2.py/.md       # sibling project: GmanCoatV1.17 coating sim
├── referencecode/        # earlier reference snapshots
├── image.png             # ring-clutch transmission design reference
└── Plan.md · plan2.md · "Goal of GmansRunV1.17" · MODEL_REFINEMENT.md
                          # historical notes (executable constants win)
```

## Testing / Verification

```bash
python _smoke.py      # boots headless, exercises all 3 modes, gear seating,
                      # clutch-slip sweep, tab clicks; prints per-mode fps
python _mpgtest.py    # 600-s cruises; asserts stored energy never rises
                      # without fuel (conservation check)
python optimize_control.py           # re-verifies firing point + SOC policy
python optimize_harvest.py 300000    # re-verifies component sizing
```

Latest optimizer verification (this build): firing point = optimum
(400 rpm × 8/rev, +0.0%), sizing = optimum (+0.0% cycle MPG over ~4.3k
re-evaluated configs), SOC policy updated to on 0.22 / off 0.42 / max 0.52
(~770 MPG demanding cycle, engine-on 17.0%).

## Performance Architecture (why it's smooth now)

The numpy/pygame software renderer was profiled and restructured end-to-end:

- **Per-mesh transform cache** — world+camera verts reused while nothing
  moved; `spin == 0` meshes cache across frames even while the engine spins.
- **Per-mesh shading cache** — an unchanged mesh reuses its entire polygon
  contribution (this is why CAR VIEW roughly tripled in fps).
- **Fully vectorized** culling (near-plane, screen bounds, ≥3 px² sub-pixel,
  backface), studio shading and depth fog; Python only touches drawn faces.
- **One-shot point lists** — a single `tolist()` per mesh builds every face.
- **Draw-loop LOD** — fill-only below 60 px², AA edges above, facet outlines
  ≥140 px², panel-gap tint ≥320 px².
- **Adaptive quality** — AA while fps ≥ 45; below 30 the model renders at
  0.66×/0.5× and smoothscales — but still **re-renders every frame**, so motion
  never freezes; drive-mode car + inset amortize every other frame.

Measured headless: PREVIEW ~4 → ~7 fps, CAR VIEW ~8 → ~22 fps at max quality;
the auto-degraded path is proportionally faster still.

## Honest Framing

This is a first-order conceptual engineering simulator — a thinking tool, not
a validation package. MPG figures are model outputs under stated assumptions;
"solar-sustained" points mean external solar covers the road load at that
instant (not free energy); optimizer results hold only within their defined
bounds; and the renderer is for explanation, not CAD export. See
**[overview.md](overview.md)** for the full breakdown.
