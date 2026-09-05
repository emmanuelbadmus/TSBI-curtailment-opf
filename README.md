# Two-stage bidirectional inverter in an AC distribution OPF

An equivalent circuit model of a two-stage bidirectional inverter (TSBI) [1],
solved jointly with the distribution network it is connected to. The inverter
carries its converter stages, semiconductor losses and LCL filter explicitly
rather than a fitted efficiency curve, so its operating point and its coupling
to the grid are found by the same optimization that solves the network.

PV curtailment is the study demonstrated here, following ANOCA [2]: feeders
are converted from OpenDSS to BMOPF JSON with PowerIO [3], an inverter is
connected at chosen terminals, and each case is solved as an AC optimization
with hard network and inverter limits.

120 cases: `2 active-power controls × 4 reactive-power controls × 3 objective norms × 5 feeders`

- Active power: `CONSTANT_P`, `MPPT`
- Reactive power: `UPF`, `CONSTANT_Q`, `CPF`, `VOLT_VAR`
- Objective norm: `L1`, `L2`, `LINF`

Each scenario JSON carries what varies between scenarios: network, PV
single-diode model, inverter, controls, objective, line ratings and solver
settings. What belongs to the feeder as a whole -- where its inverters go and
the voltage band it is held to -- lives once in `data/dss/<case>/study.json`,
and every published result records the band it was solved under.

The PV block carries both the single-diode parameters and the module's
datasheet `v_mp_v` and `i_mp_a`. The diode curve is what the optimization
solves, and a fitted single-diode model does not reproduce the datasheet
maximum exactly: here the curve peaks at `4880.0 W` per inverter against the
`4800.8 W` the datasheet pair multiplies to, 1.7% apart. Every published
figure uses the curve.

## Connecting an inverter

An inverter is attached to one phase terminal. Three lines in the scenario
JSON say which type, which PV array, and where:

```json
"inverter_connections": {
  "inverter_4a": {"id": "inverter_1", "dc_src_id": "pv_1", "terminal": "n4.1"}
}
```

`id` names an entry in the `inverter` section, which states the hardware once
and is shared by every connection: apparent-power rating, DC-link voltage,
switching device parameters, LCL filter, and the active and reactive power
control. `dc_src_id` names an entry in `pv`, a single-diode array. `terminal`
is a bus and phase already in the network.

Nothing else has to change. The parser builds one `Inverter` per connection,
and it enters the same phase-domain formulation as every line and load.

## How it enters the OPF

Each connected inverter contributes its own equations, solved simultaneously
with the network rather than after it:

| Stage | States |
| --- | --- |
| PV array | the single-diode current-voltage relation, or a held voltage under `CONSTANT_P` |
| First stage (DC-DC) | the boost voltage relation and its power balance, with switching and conduction losses |
| Second stage (DC-AC) | the modulation limit, the AC-side voltage in real and imaginary parts, and its power balance |
| LCL filter | Kirchhoff's current and voltage laws across the filter |
| Control | the active and reactive power law of the selected mode |
| Interface | the real and imaginary current injected into the terminal |

That current is what the network sees: it enters the KCL row of the terminal
alongside the lines, loads and shunts already there, so the inverter's
operating point and the feeder's voltages are one solution, not two.

The optimization then chooses how much of the available PV power to give up:

```
minimise  ‖ curtailment ‖        over L1, L2 or L-infinity
subject to  the network and inverter equations above,
            bus voltages within their band, relative to the no-PV base case,
            line currents within their ratings,
            each inverter within its kVA rating.
```

The three norms answer different questions of the same feeder: `L1` the least
total energy given up, `L2` the least given up while spreading it, `LINF` the
least asked of the worst-affected inverter.

## Quick start

The study needs Ipopt, a native solver binary pip cannot install, so it runs in
a conda environment. If you do not have conda, install
[Miniforge](https://github.com/conda-forge/miniforge) first.

```bash
git clone --depth 1 --single-branch https://github.com/emmanuelbadmus/inverter-distribution-opf.git
cd inverter-distribution-opf

conda env create -f environment.yml       # mamba and micromamba work too
conda activate inverter-distribution-opf  # <- required; see below

python run_curtailment_study.py --check            # confirms the setup
python run_curtailment_study.py --case 4bus_120v   # one feeder, ~10 s
python run_curtailment_study.py                    # all 120 cases
```

**`conda activate inverter-distribution-opf` is the step that gets
skipped.** A prompt reading `(base)` means it did not happen, and the
environment you are in has none of this. `--check` reports exactly what is
present and what it is missing, and is worth running before a session rather
than mid-command.

The converted feeder networks are tracked, so solving needs only Pyomo, Ipopt
and numpy. The 120 scenarios are generated from those networks on first run,
in about a second, using nothing but the standard library. Rebuilding a network
from its OpenDSS source is the only step that needs OpenDSS and powerio, and it
is needed only after changing a feeder.

Solves use every core by default, so `--workers` is only worth setting to hold
some back (`--workers 4`) or to make a run reproducible.

## The pipeline

```bash
python converters/dss_to_bmopf.py                    # data/dss/<case>/network/Master.dss
                                          #   -> data/bmopf_json/<case>/network/bmopf.json
python converters/generate_curtailment_scenarios.py  #   -> <case>/scenarios/*.json  (24 per feeder)
python run_curtailment_study.py           #   -> results/curtailment_matrix.json
                                          #   -> results/curtailment_by_mode.png
python converters/validate_against_opendss.py   # checks the conversion against OpenDSS
```

The feeders in `data/dss/` and the networks converted from them are both
tracked; the 120 scenarios are not. `run_curtailment_study.py` builds those on
first use, so a clone needs no separate step, and the first stage above is
needed only after changing a feeder. Run the two conversion stages together
when you do: a
scenario carries its own copy of the network and records which network it was
built from, so converting again without regenerating the scenarios is refused
rather than solved as a stale feeder.

The first two stages only need rerunning when a feeder or a study choice
changes.

Filters run one feeder or control combination. A run writes to
`results/curtailment_matrix.json` unless told otherwise, so pass `--out` for a
partial run or it overwrites the published 120-case matrix:

```bash
python run_curtailment_study.py --case M1 --norm L2 --out results/m1_l2.json
```

`--plot-only` re-renders the heatmap from an existing matrix without solving.
`--tee` streams the Ipopt log to the console, on one worker so parallel solves
cannot interleave their output. `--no-warm-start` drops the reference solve's
voltages as the initial point. The feasible set does not depend on them, but
the problem is not convex and Ipopt reports a local optimum: 49 of the 120
cases then fail to converge, and of the 71 that still solve, 24 land on a
worse point than the warm start finds. The published run is warm-started.

## Reproducibility

The published run was made in the environment `environment.yml` describes:
Python 3.12.14, Ipopt 3.14.20, Pyomo 6.10.1, NumPy 2.3.5, powerio 0.8.3, on
macOS arm64. Solving the same 120 cases under Python 3.14 moves one M1 case by
`9e-6` of its value and leaves the other 119 identical, which is the size of
difference to expect from a different build rather than a different model.

A run writes three files, all from the same solve:

| File | Contents |
| --- | --- |
| `results/curtailment_matrix.json` | the full result, per case |
| `results/runtime_table.csv` | one row per case: termination, iterations, time, solver settings, residuals |
| `results/curtailment_by_mode.png` | the heatmap |

The matrix records what produced it: the Ipopt build, the Pyomo, numpy and
powerio versions, the platform and core count, the thread limits, the per-unit
base, and `code_sha256`, a hash of the solver and runner sources. Recompute
that hash in any checkout and compare — it identifies the code even where a
commit hash would not, such as results republished into a fresh history.

Each case additionally carries its own solver settings, termination status,
iteration count and final residuals.

`--tee` streams the Ipopt log to the console, on one worker so parallel solves
cannot interleave:

```bash
python run_curtailment_study.py --case M1 --p-mode MPPT --norm L2 --tee
```

`--solver` picks Ipopt's linear solver. The default is `mumps`, which every
Ipopt build carries and which the published results use; `ma27`, `ma57`,
`ma77`, `ma86` and `ma97` need an Ipopt linked against HSL, and `pardiso`,
`pardisomkl`, `spral` and `wsmp` a build carrying those. All of them are
loaded at run time, so `--check` lists the ones a machine actually has and a
run refuses to start on one it does not:

```bash
python run_curtailment_study.py --case M1 --norm LINF --solver ma57
```

The linear solver changes how the KKT systems are factorised, not what is
being solved. Every published row records the one it was solved with, and
`mumps_pivot_tolerance` is recorded only when MUMPS was in use, since it is a
MUMPS option and is not sent to the others.

All six available here solve all 120 cases, and agree on the norms whose total
is pinned to within `4e-05` relative:

| `--solver` | solved | agreement with MUMPS, `L1` and `L2` | `LINF` |
| --- | --- | --- | --- |
| `mumps` | 120/120 | -- | -- |
| `ma27` | 120/120 | `3.8e-05` | `4.1e-05` |
| `ma57` | 120/120 | `2.9e-05` | `8.1e-04` |
| `ma77` | 120/120 | `4.9e-07` | `2.6e-04` |
| `ma86` | 120/120 | `1.6e-08` | `2.6e-04` |
| `ma97` | 120/120 | `1.5e-06` | `6.8e-04` |

The wider `LINF` column is the same effect noted above: minimising the largest
per-inverter curtailment does not pin the total, so two equally optimal
factorisations can distribute it differently. `LINF` peaks agree to `8e-04`.

`ma77` is out-of-core and keeps its factorisation in scratch files named the
same way for every solve, so each `ma77` solve is given a temporary directory
of its own. Without that, parallel solves overwrite each other -- 18 of 24
`4bus` cases failed -- and an interrupted run leaves the files in the
repository.

## Layout

```
run_curtailment_study.py           the pipeline, in order: inputs, solve, publish
converters/                        builds the study inputs from the feeders
  dss_to_bmopf.py                  OpenDSS -> BMOPF networks (powerio)
  opendss_electrics.py             line, load, shunt and source data read
                                   back from OpenDSS
  generate_curtailment_scenarios.py
                                   FeederLibrary: the feeders on disk and the
                                   study design of each
  validate_against_opendss.py      power-flow agreement with OpenDSS
src/                               the solver
  bmopf_parser.py                  BMOPFParser: JSON -> object model
                                   StudyInputs: find and build the scenarios
  network_loader.py                DxNetworkModel: the network container
  network_simulator.py             per-unit helpers, the Pyomo formulation,
                                   ScenarioSolver: one scenario, two stages
                                   CurtailmentStudy: the sweep, run and publish
  network_plotter.py               NetworkPlotter: the table and the heatmap
  provenance.py                    RunProvenance: what produced a result
  components/                      bus, line, transformer, switch, load, shunt,
                                   slack, generator, inverter, pv
data/dss/<case>/network/           OpenDSS feeder input (tracked)
data/dss/<case>/study.json         optional: where that feeder's inverters go
data/bmopf_json/                   generated: converted networks + scenarios
results/                           published study outputs (matrix + heatmap)
environment.yml                    the conda environment, incl. Ipopt
```

## Feeders

Tracked as OpenDSS text in `data/dss/<case>/network/Master.dss`. Cite the
original sources when publishing results derived from them.

| Case | Source | Notes |
| --- | --- | --- |
| `4bus_120v` | Authored here | IEEE 4-bus Y-Y test feeder, 208 V secondary, three inverters |
| `P` | Not recorded in the feeder files | 11 kV radial, 0.4/0.415 kV secondaries |
| `M1` | EPRI Feeder M1, dpv.epri.com, 2013 | DPV Monitoring and Feeder Analysis (P174) |
| `J1` | EPRI Feeder J1, dpv.epri.com, 2013 | DPV Monitoring and Feeder Analysis (P174) |
| `p5rhs0_1247--p5rdt52` | NREL SMART-DS, inferred from the name | synthetic 12.47 kV feeder |

To add one, put its OpenDSS files under `data/dss/<case>/network/` and rerun
the pipeline. Nothing else is needed: the feeders are discovered on disk, so
no list in the source has to be told about them, and a feeder with no study
design of its own is swept like any other — an inverter at every terminal
carrying load, and the default voltage band.

To place the inverters differently, or to hold a feeder to a different band,
put a `data/dss/<case>/study.json` beside it:

```json
{
  "label": "My feeder",
  "terminal_selection": "top_positive_load_terminals",
  "selection_count": 100,
  "repeats": 2,
  "voltage_magnitude_limits_relative_to_base_case": {
    "minimum": 0.95, "maximum": 1.05
  }
}
```

Every field is optional. `terminals` states an explicit list instead of a
selection, and `groups` states several placements at once; the five feeders
here each carry one of these files, so they double as worked examples.

## Study choices

| Setting | Value | Note |
| --- | --- | --- |
| Constant-Q setpoint | `-0.05 pu`, i.e. `-290 var` per 5.8 kVA inverter | absorption; a mild explicit baseline, not an IEEE 1547 [4] default |
| Constant power factor | `+0.95` | positive is injection in this model's sign convention |
| Voltage band | `0.95`-`1.05` of the base case | 4bus uses `0.95`-`1.005` so voltage-driven curtailment binds |

`LINF` minimises the largest weighted per-inverter curtailment and pins
nothing else, so a LINF case's *total* curtailment can differ between runs
that are equally optimal. Compare LINF cases on `max_w`.

The four reactive-power modes are the smart-inverter functions IEEE 1547
defines and [5] surveys; [6] reports hardware-in-the-loop measurements of an
inverter operating them.

## Code checks

```bash
ruff check .
ruff format --check .
```

Rules live in `pyproject.toml`; ruff is part of the environment.

## Other platforms

Development happens on macOS, so `.github/workflows/cross-platform.yml` builds
the environment and runs the study on Linux and Windows rather than assuming
what one machine measured travels. It builds the study inputs from the feeders
first and checks the conversion against OpenDSS on each platform, because that
conversion reads the feeder's electrics out of a native OpenDSS binary that
differs by platform.

It is a smoke test, not the published run: it lints, builds the inputs, solves
the L2 cases of `4bus_120v` and `J1`, and checks two feeders against OpenDSS.
The 120-case matrix in `results/` is produced locally in the environment
`environment.yml` describes. Run it after changing `environment.yml` or a
feeder. It has caught real faults: a feeder whose master file named two of its
own includes by a spelling the files do not have, which only a case-sensitive
filesystem notices, and console output the default Windows codec could not
encode.

## Agreement with OpenDSS

The study's own checks show a solution is self-consistent; they cannot show the
feeder was translated correctly. `converters/validate_against_opendss.py` does
that, solving each feeder in both tools without PV and comparing every bus:

```bash
python converters/validate_against_opendss.py
python converters/validate_against_opendss.py --case M1 --tolerance 1e-3
```

Voltages are compared as phasors in volts, not per unit: the two tools need not
pick the same base. Magnitude and angle both count, after removing one global
rotation, so a wrong transformer phase shift or a swapped phase shows up. A
node OpenDSS energises that this model lacks fails the comparison, so agreement
cannot be reached over a subset.

| Feeder | Nodes | Mean error | Max error |
| --- | --- | --- | --- |
| `4bus_120v` | 12 | 1.5e-8 | 3.1e-8 |
| `P` | 368 | 2.9e-3 | 7.3e-3 |
| `p5rhs0_1247--p5rdt52` | 492 | 5.7e-4 | 8.3e-4 |
| `M1` | 3153 | 1.4e-4 | 7.5e-4 |
| `J1` | 4245 | 9.5e-4 | 2.0e-3 |

The table is also the regression bar. A fixed tolerance only catches an error
large enough to reach it: scaling every line impedance on `4bus_120v` by 1.5
moves its worst bus only `2.6e-3`, inside one percent. A conversion that gets
more than twice as far from OpenDSS as the figure above fails as a regression,
and a failing run does not overwrite the figure it was measured against.
`--no-baseline` turns that off.

The only nodes here and not in OpenDSS are the source Thevenin terminals,
which OpenDSS keeps inside its `Vsource` object; anything else extra fails.

### Why the converter reads OpenDSS back

`powerio` carries the topology faithfully but not every electrical quantity, so
`converters/opendss_electrics.py` re-reads these from OpenDSS. Each was found
by comparing the two power flows, and each moved the agreement above:

| Quantity | What the converter alone produced |
| --- | --- |
| Line series impedance | per-length matrices in whichever unit the source file used, and re-derived rather than copied for `linegeometry` lines |
| Line phasing | the full linecode on lines OpenDSS Kron-reduces onto fewer phases, which also invented bus terminals |
| Line charging | susceptance evaluated at 50 Hz on 60 Hz feeders |
| Load model | every load as constant power, dropping the CVR voltage dependence on `M1` and `J1` |
| Capacitors | full nameplate susceptance whether or not the bank is switched in |
| Switch impedance | nothing, leaving a closed switch an ideal short |
| Source | an ideal terminal voltage, with no Thevenin impedance behind it |
| Out-of-service branches | lines and switches OpenDSS holds disabled were still energised |

Line impedances are written as absolute ohms from OpenDSS's own matrices,
removing every unit and reduction assumption from the pipeline.

## Conversion warnings

`converters/dss_to_bmopf.py` warns about OpenDSS fields BMOPF cannot represent. Monitors,
`WireData`/`LineGeometry`/`LineSpacing` definitions, transformer thermal
ratings and wye-wye phase decomposition are expected and do not change the
electrical solution. A warning that a line, load, shunt, switch, voltage
source or transformer was *dropped* does need investigation. `CapControl` and
`RegControl` are not simulated, so this is a static study: capacitor banks are
taken in the switching state OpenDSS settles on, and regulators at their taps
there.

## Solver settings

Every case is solved with identical settings. Nothing is tuned per feeder,
control mode or norm, so no case is held to a different standard than another:

| Setting | Value |
| --- | --- |
| `linear_solver` | `mumps`, overridable with `--solver` |
| `mumps_pivot_tolerance` | `1e-3`, sent only with MUMPS |
| `tol` | `1e-9` |
| `constraint_violation_tolerance` | `1e-5` |
| `acceptable_tol` / `acceptable_constr_viol_tol` / `acceptable_iter` | `1e-6` / `1e-4` / `20` |
| `initial_delivery_fraction` | `1e-4` |
| `mu_init` / `mu_strategy` / `retry_mu_strategy` | `0.1` / `adaptive` / `monotone` |
| `max_iter` / `timeout_s` | `5000` / `60` CPU seconds |

`mumps_pivot_tolerance` is the one value chosen rather than left at its
default: Ipopt's `1e-6` is too permissive for these KKT systems and `1e-2` too
aggressive, each failing cases the other solves. `1e-3` was the only value in
the sweep that solved all 120.

`acceptable_constr_viol_tol` is Ipopt's fallback termination threshold, not
this study's acceptance criterion; the post-solve check below decides that, and
does not consult it. It certifies feasibility, not global optimality: the
formulation is not convex, so curtailment figures are upper bounds on what the
feeder requires, not proven minima.

### What is enforced, and how a case is accepted

Each solved case is re-checked outside the solver: KCL residuals, the voltage
band, line ampacity, inverter apparent power and variable bounds. A constraint
passes when its violation is within the absolute tolerance,
`max(5 * constraint_violation_tolerance, 5e-5)`, **or** within `1e-14` of the
largest term the constraint sums, whichever is larger.

The relative allowance covers rows whose arithmetic cannot carry the fixed
bar: cancelling terms of order `1e11` in double precision leaves about `2e-5`
of unavoidable rounding. No case in the published 120 reaches it — the worst
residual anywhere is `1.0e-07` — so every case is accepted on the absolute bar
alone.

### Model regularisation

No electrical quantity is substituted when a feeder omits it; that is an
error, not something to fill in. What remains are numerical guards, all in per
unit so the same negligible fraction applies at every voltage level:

| Guard | Value | Why | Activates |
| --- | --- | --- | --- |
| Bus shunt conductance | `1e-9` pu on every bus | keeps an otherwise floating bus from making the system singular | every bus, by construction |
| Transformer series floor | `1e-8` pu resistance, `1e-6` pu reactance | a winding that declares zero impedance, as an ideal regulator does, is not otherwise representable | never on these five feeders |
| `loss_smoothing_epsilon` | `1e-8` | keeps the inverter loss terms differentiable near zero current | inverter loss terms |

A load OpenDSS holds outside its `Vminpu`-`Vmaxpu` band becomes a constant
impedance. That band is the load model's own switching threshold, read from
the feeder, and is not the study's voltage constraint above: it applies only
where a load sits, and in absolute per unit of that load's nominal voltage.
Which band it falls in is a discrete choice, fixed before each solve
rather than switched on a variable, which would make the problem non-smooth. A
flat start would misplace every load that ends up outside the band, so the
reference power flow repeats with the choice re-taken at its own solution until
nothing moves — two or three passes on the two feeders where any load leaves
the band. The curtailment optimization then holds those bands, which is the
operating point OpenDSS is compared against. A load within `2e-5` pu of an edge
stays inside, so rounding alone cannot move it between passes.

A reference power flow that does not converge is retried once from the same
initial point with a small objective pulling the voltages toward it. The
constraints are unchanged, and no published case reaches it.

Each case is solved in two stages: a reference power flow without PV, which
sets the relative voltage band and the initial point, then the curtailment
optimization. `stages_per_case` in the matrix counts those two stages, not
solver invocations: on a feeder with CVR loads the reference stage repeats
until the voltage bands settle, so Ipopt is called three or four times. `--no-warm-start` drops the second use of that reference; 49 of
the 120 then fail to converge and 24 more land on a worse local optimum.

## Licence

The code is MIT licensed; see `LICENSE`. The feeders under `data/dss/` are not
covered by it and keep the terms of whoever published them: `M1` and `J1` are
EPRI's, released publicly under its Distributed Renewables Research Program
(P174), and their master files carry the citation EPRI asks for. Cite the
original sources when publishing results derived from them.

## References

[1] E. O. Badmus and A. Pandey, "Two-stage bidirectional inverter equivalent
circuit model for distribution grid steady-state analysis and optimization,"
*IEEE Trans. Power Syst.*, 2026.

[2] E. O. Badmus and A. Pandey, "ANOCA: AC network-aware optimal curtailment
approach for dynamic hosting capacity," in *Proc. 63rd IEEE Conf. Decision and
Control (CDC)*, Milan, Italy, 2024, pp. 5338–5345, doi:
[10.1109/CDC56724.2024.10886763](https://doi.org/10.1109/CDC56724.2024.10886763).

[3] Eigenergy, *PowerIO: Case parsing, conversion, matrices, and language
bindings for power system data*, version 0.8.3. [Online]. Available:
https://github.com/eigenergy/powerio (accessed Sep. 9, 2026).

[4] *IEEE Standard for Interconnection and Interoperability of Distributed
Energy Resources with Associated Electric Power Systems Interfaces*, IEEE Std
1547-2018, 2018. [Online]. Available:
https://standards.ieee.org/ieee/1547/10906/

[5] *Impact of IEEE 1547 Standard on Smart Inverters and the Applications in
Power Systems*, IEEE Power & Energy Society, Tech. Rep. PES-TR67, 2020.
[Online]. Available:
https://www.nlr.gov/media/docs/libraries/grid/smart-inverters-applications-in-power-systems.pdf

[6] National Renewable Energy Laboratory, *500 kW Photovoltaic Inverter
Hardware-in-the-Loop Testing*, Golden, CO, USA, Tech. Rep. NREL/CP-5500-56833,
2013. [Online]. Available: https://docs.nlr.gov/docs/fy13osti/56833.pdf
