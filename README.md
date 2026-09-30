# PARL-ID: Physics-Informed Adjoint Inverse Design with RL in the Fabrication Loop

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21290617.svg)](https://doi.org/10.5281/zenodo.21290617)

Python code for the manuscript **"PARL-ID: A Fabrication-Aware Physics-Informed
Adjoint Framework With Reinforcement Learning in the Loop for Inverse Design of
CMUT and Photonic Sensors"** (in preparation for IEEE Sensors Journal)
by Tanvir M. Mahim, M. Mosaddequr Rahman, and A.H.M.A. Rahim
(BRAC University).

It builds on the authors' earlier paper in IEEE Sensors Journal 25(13),
26104-26116 (2025), https://doi.org/10.1109/JSEN.2025.3569424.

- Repository: https://github.com/Tanvir-Mahmud-Mahim/Fabrication-Aware-Physics-Informed-Adjoint-Framework-With-RL-in-the-Loop-for-Inverse-Design-of-CMUT
- CMUT simulation database (369 designs x 150 frequencies = 55,350 rows): https://doi.org/10.5281/zenodo.21290617

The manuscript itself (LaTeX sources) is not part of this repository.

---

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [Installation](#3-installation)
4. [Quick start: three ways to use the code](#4-quick-start-three-ways-to-use-the-code)
5. [The scripts, step by step](#5-the-scripts-step-by-step)
6. [Which script makes which figure](#6-which-script-makes-which-figure)
7. [The Python modules](#7-the-python-modules)
8. [Where the numbers come from](#8-where-the-numbers-come-from)
9. [Built-in checks](#9-built-in-checks)
10. [Notes on the calculations](#10-notes-on-the-calculations)
11. [Version history](#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

---

## 1. The idea in one minute

A **CMUT** (capacitive micromachined ultrasonic transducer) is a tiny drum:
a thin membrane held over a narrow gap, moved by a voltage to send and
receive ultrasound. How strongly it moves at a given frequency depends on the
thicknesses of its layers. Here the design has four numbers, all in
micrometres: the electrode thickness `t_e`, the nitride layer thickness
`t_np`, the oxide thickness `t_ox`, and the support-wall thickness `t_w`.

The goal is **inverse design**: choose those four numbers so that the membrane
moves as much as possible at a target frequency (4.3 MHz by default), and
keep the design good even when the real fabrication process makes every layer
slightly wrong. The code does this in three stages:

1. **A fast stand-in model (surrogate).** A neural network is trained on
   finite-element simulation results. In the PyTorch code it is a
   **physics-informed neural network (PINN)**: its training also penalises
   breaking the plate equation of the membrane, not only mismatch with data.
2. **Gradient-based design (the "adjoint" step).** Because the surrogate is
   differentiable, the code can compute how the objective changes with each
   design number and walk uphill. ("Adjoint" is the standard name for this
   efficient way of getting gradients.)
3. **Fabrication loop with reinforcement learning (RL).** A simulated
   fabrication step randomly perturbs the design (film thickness, etching,
   stress). An RL agent learns a small correction that improves the
   **worst-case** outcomes. The default score is **CVaR** (conditional
   value-at-risk): the average of the worst 5 % of the perturbed results,
   and never fewer than one result. With the RL environment's default of 16
   random draws, 5 % rounds up to one draw, so the score is simply the
   single worst draw (see Section 10).

A second, simpler problem (light travelling in a silicon waveguide, solved
with a small finite-difference solver) is used to check the gradient
machinery against an exact answer.

---

## 2. What is in this repository

```
Fabrication-Aware-Physics-Informed-Adjoint-Framework-With-RL-in-the-Loop-for-Inverse-Design-of-CMUT/
|-- README.md               this guide
|-- CHANGELOG.md            what changed, newest first
|-- CITATION.cff            citation details (drives the "Cite this repository" button)
|-- LICENSE                 Apache-2.0 license
|-- requirements.txt        Python packages to install
|-- parl_id/                the Python package
|   |-- physics/            plate equation, waveguide (Helmholtz) solver, acoustic loading
|   |-- pinn/               neural network, loss function, training loop
|   |-- adjoint/            gradient-based design optimiser, fabrication "snapping" helpers
|   |-- rl/                 fabrication environment, GCN-SAC agent, PPO agent, evolutionary warm start
|   `-- data/               loader for the CMUT database (+ synthetic stand-in), waveguide data
|-- experiments/
|   |-- run_cmut_pinn.py        stage 1: train the CMUT PINN
|   |-- run_adjoint_design.py   stage 2: gradient design through the trained PINN
|   |-- run_rl_fab_loop.py      stage 3: RL in the simulated fabrication loop
|   |-- run_photonic_pinn.py    waveguide test problem: PINN + gradient check
|   `-- make_figures/           numpy/scipy scripts that draw the result figures (no PyTorch)
|       |-- common.py               shared style, file paths, objective helpers
|       |-- figA.py                 Fig_pinn_train.svg
|       |-- figB.py                 Fig_field.svg, Fig_grad.svg
|       |-- figC1.py                Fig_inv.svg, Fig_yield.svg
|       |-- cvar_study.py           CVaR correction study (printed table)
|       |-- figC2.py                Fig_transfer.svg
|       |-- extensions_study.py     warm start, bandwidth objective, ranking quality (printed)
|       |-- fig_ablation.py         Fig_ablation.svg
|       `-- deck.js                 builds an editable PowerPoint of the figures (Node.js)
`-- tests/
    `-- test_smoke.py       eight quick checks that every part runs
```

**The database is not stored in this repository.** Download it from Zenodo
(see [Way B](#way-b-redraw-the-result-figures-from-the-database)).
Scripts write results to `outputs/` (PyTorch stages; the figure scripts
write their SVG files to `outputs/figures/`) and to
`experiments/make_figures/` (small `.npy`/`.npz` state files); all of these
are ignored by git. No script writes outside the repository.

---

## 3. Installation

The repository does not state a minimum Python version. The checks in this
guide used **Python 3.11**.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

`requirements.txt` installs `torch` (2.3 or newer; 2.4.1 or newer on
Windows), `numpy`, `scipy`, `pandas`, `matplotlib`, and `gymnasium` (the
standard interface for RL environments). PyTorch is only needed for the PINN/adjoint/RL stages and the
tests; the figure scripts need only `numpy`, `scipy`, `pandas`, and
`matplotlib`. If you want a smaller download without GPU support, pick the
CPU build on the PyTorch website before running the line above.

Optional extras (not needed by any script listed below):

- `ceviche-challenges`: used only by `make_mode_converter()` in
  `parl_id/data/ceviche_loader.py`; no experiment script calls it.
- `stable-baselines3`: mentioned in `requirements.txt` as an alternative
  to the bundled PPO; no code in this repository imports it.
- `pytest`: only if you prefer `python -m pytest tests/ -q` to the plain
  command in Way A.

**Why PyTorch 2.3.** The code itself runs with NumPy 1.24 and with NumPy 2
(it uses no function that exists in only one of them), and pip may install
NumPy 2. PyTorch releases before 2.3 were built for NumPy 1 only: with
NumPy 2, `torch.from_numpy` stops with "Numpy is not available" (checked
here with PyTorch 2.2.2 and NumPy 2.0.0). On Windows this was fixed only in
PyTorch 2.4.1. The other minimums are unchanged. SciPy releases before
1.13 declare a NumPy limit below 2, so pip never pairs them with NumPy 2,
and Gymnasium 0.29.0 ran the tests here with NumPy 2.0.0. One caution:
pandas 2.0.0 to 2.1.1 and Matplotlib 3.7.0 to 3.7.2 were built for NumPy 1
but do not say so in their package data, so pip can install them next to
NumPy 2, and then they fail on import (checked here: pandas 2.0.0 with
NumPy 2.0.0 stops with "numpy.dtype size changed", and Matplotlib 3.7.0
with NumPy 2.0.0 cannot import `matplotlib.pyplot`). A fresh
`pip install -r requirements.txt` installs current versions and is not
affected; if you pin one of those older releases, also pin `numpy<2`.

**Checked with the minimum versions** (30 September 2026, Python 3.11,
CPU): with torch 2.3.0, numpy 1.24.0, scipy 1.10.0, pandas 2.0.0,
matplotlib 3.7.0 and gymnasium 0.29.0, and again with torch 2.3.0,
numpy 2.0.0, scipy 1.13.0, pandas 2.2.2, matplotlib 3.8.4 and
gymnasium 0.29.0, the smoke tests passed, the three PyTorch stages ran with
short settings (`--epochs 5`, `--iters 5`, `--rl-iters 1`), and the figure
scripts ran on a stand-in database file (the real one could not be
downloaded here). The saved `.npz` state files of the two set-ups agreed to
within 2e-13.

The scripts use a CUDA GPU automatically when PyTorch finds one (the RL loop
always runs on the CPU). Everything also runs on a CPU.

---

## 4. Quick start: three ways to use the code

Run all commands from the repository folder unless a step says otherwise.

### Way A: check that everything works (under half a minute)

```
python tests/test_smoke.py
```

Each of the eight tests prints `PASS ...`, and the last line reads
`all smoke tests passed`. (A PyTorch `UserWarning` about converting a tensor
to a number may appear; it is harmless.)

### Way B: redraw the result figures from the database

No PyTorch is needed.

1. Download `cmut_inverse_design_benchmark.csv` from
   https://doi.org/10.5281/zenodo.21290617 and save it as
   `experiments/augmented_reshaped_dataset.csv`.
2. Run the scripts in this order (later ones read files written by earlier
   ones):

   ```
   cd experiments/make_figures
   python figA.py
   python figB.py
   python figC1.py
   python cvar_study.py
   python figC1.py
   python figC2.py
   python extensions_study.py
   python fig_ablation.py
   ```

   `figC1.py` runs twice: the second run picks up `cvar_state.npz` from
   `cvar_study.py` and adds the third (CVaR) curve to `Fig_yield.svg`.

### Way C: run the PyTorch pipeline

```
python experiments/run_cmut_pinn.py --csv experiments/augmented_reshaped_dataset.csv
python experiments/run_adjoint_design.py --target-freq 4.3
python experiments/run_rl_fab_loop.py
```

Leave out `--csv` to train on a built-in synthetic stand-in database instead
of the real one. The second script needs `outputs/cmut_pinn.pt` from the
first; the third needs both `outputs/cmut_pinn.pt` and
`outputs/adjoint_design.pt`. The first and third are long at their default
settings; add `--epochs 5` and `--rl-iters 1` for a quick trial run.

---

## 5. The scripts, step by step

### PyTorch stages (`experiments/`)

| Command | What it does | Time* | Results |
|---|---|---|---|
| `python tests/test_smoke.py` | Eight quick checks (Section 9) | 7 s (21 s under load) | printed |
| `python experiments/run_cmut_pinn.py [--csv FILE] [--epochs N]` | Trains the CMUT PINN (default 2000 epochs) on the database, or on the synthetic stand-in without `--csv` | default: long (not re-timed); `--epochs 5` (synthetic data): 9 s (78 s under load) | `outputs/cmut_pinn.pt` |
| `python experiments/run_adjoint_design.py [--target-freq 4.3] [--iters 200]` | Freezes the PINN and runs gradient ascent (Adam, 4 random restarts) on the average membrane displacement at the target frequency, inside the design ranges | default (`--iters 200`): 7 s (57 s under load); `--iters 5`: 6 s | `outputs/adjoint_design.pt`; prints the best design |
| `python experiments/run_rl_fab_loop.py` | Scores the nominal design under random fabrication errors, then trains the RL agent to correct it and scores the corrected design | default (`--rl-iters 20`): long (not re-timed); `--rl-iters 1`: 9 s; `--rl-iters 2`: 12 s (85 s under load); `--agent ppo --rl-iters 1`: 11 s (49 s under load) | `outputs/rl_fab_loop.npz` |
| `python experiments/run_photonic_pinn.py [--epochs 1500]` | Waveguide test problem: finite-difference reference fields, a Helmholtz PINN fitted to them, and a gradient check (see the caveat in Section 10) | default: long (not re-timed); `--epochs 5`: 6 s (8 s under load) | `outputs/photonic_pinn.pt`, `outputs/photonic_ref.npz` |

Options of `run_rl_fab_loop.py` (all used in the ablations):

```
python experiments/run_rl_fab_loop.py                     # GCN-SAC agent + CVaR score + evolutionary warm start (default)
python experiments/run_rl_fab_loop.py --agent ppo         # PPO agent instead
python experiments/run_rl_fab_loop.py --reward mean_std   # score = mean - 1 x standard deviation instead of CVaR
python experiments/run_rl_fab_loop.py --no-evo-warmstart  # skip the evolutionary warm start
python experiments/run_rl_fab_loop.py --rl-iters 20       # training length: PPO iterations of 256 steps, or 100 x this many GCN-SAC steps
```

### Figure scripts (`experiments/make_figures/`, numpy/scipy only)

| Command | What it does | Reads | Writes |
|---|---|---|---|
| `python figA.py` | Trains a small numpy neural network on the database with 20 % of the designs held out; convergence curves and held-out parity | database | `Fig_pinn_train.svg`, `r2.npy` |
| `python figB.py` | Finite-difference waveguide field; exact adjoint gradient compared with central finite differences at 40 random pixels | database (only for panel (a) of `Fig_field`) | `Fig_field.svg`, `Fig_grad.svg`, `cos.npy` |
| `python figC1.py` | Inverse design at 4.3 MHz (200 random probes + gradient refinement of the best 4) versus random search; yield under 1000 random fabrication errors | database, `cvar_state.npz` if present | `Fig_inv.svg`, `Fig_yield.svg`, `state_c1.npz` |
| `python cvar_study.py` | Searches for the design with the best CVaR (average of the worst 5 %: 10 of 200 draws during the search, 100 of 2000 in the comparison) and compares four designs on 2000 common random fabrication errors | database, `state_c1.npz` | `cvar_state.npz`; table printed |
| `python figC2.py` | Tests whether a correction found at 4.3 MHz still helps at 3.0 to 6.0 MHz | database, `state_c1.npz` | `Fig_transfer.svg`, `transfer.npz` |
| `python extensions_study.py` | Warm-starting from the 4.3 MHz design at 4.0/4.5/5.0 MHz; peak versus band-average objective; ranking quality of the interpolated objective (5-fold) | database, `state_c1.npz` | `extensions_state.npz`; results printed |
| `python fig_ablation.py` | Bar charts of the ablation results (values are typed into the script) | nothing | `Fig_ablation.svg` |

SVG files go to `outputs/figures/` inside the repository (`OUT` in
`experiments/make_figures/common.py`, which creates the folder); the
`.npy`/`.npz` files stay in `experiments/make_figures/`.

\*Times measured on a shared two-core computer (CPU only, Python 3.11,
PyTorch 2.14), including start-up, with exactly the arguments shown. The same
commands took several times longer when other jobs were using the machine
("under load"), so treat these times as rough. Run times of the figure scripts on the real database were not
measured, because the database could not be downloaded here; all seven
scripts were checked to run to completion on a stand-in file with the same
columns and size.

**`deck.js`** builds an editable PowerPoint of the figures with the Node.js
package `pptxgenjs` (`npm install pptxgenjs`, then
`node experiments/make_figures/deck.js`). It reads PNG copies of the figures
from `outputs/figures/<name>.png` and writes
`outputs/PARL-ID_figures.pptx`, creating `outputs/` if needed. No script in
this repository makes the PNG copies; convert the SVG files yourself.

---

## 6. Which script makes which figure

The manuscript is not in this repository, so the mapping below uses the
labels written in the code itself (and in the earlier README); they may not
match the final article numbering.

| Label in the code | Where it appears | Script |
|---|---|---|
| Experiment 1, "Fig. F2/F3" | `run_cmut_pinn.py` docstring; the earlier README adds that this run fills the physics-ablation R² in "Sec. VI-A" | `experiments/run_cmut_pinn.py` |
| Experiment 2, "Fig. F5" | `run_adjoint_design.py` docstring | `experiments/run_adjoint_design.py` |
| Experiment 3, "Fig. F7/F8" | `run_rl_fab_loop.py` docstring; the earlier README adds that it fills the GCN-SAC/PPO numbers in "Sec. VI-D" | `experiments/run_rl_fab_loop.py` |
| Experiment 4, "Fig. F4/F6" | `run_photonic_pinn.py` docstring; `gradient_fidelity()` is called the "validation figure F4 of the paper" | `experiments/run_photonic_pinn.py` |
| "paper Eq. 17" (CVaR score), "paper Eq. 16" (mean minus spread score) | `FabricationEnv.yield_reward()` in `parl_id/rl/fab_env.py`; `--reward` help text | `experiments/run_rl_fab_loop.py` |
| "paper Section VI-D, Table FOM panel C" | `cvar_study.py` docstring | `experiments/make_figures/cvar_study.py` |
| "paper Sec. VI, ablation" (earlier README: "Sec. VI-F") | `fig_ablation.py` docstring | `experiments/make_figures/fig_ablation.py` |
| "Stage 3, main agent" (earlier README: "paper Sec. V-C") | `parl_id/rl/gcn_sac.py` docstring | `experiments/run_rl_fab_loop.py` |

The SVG figure files are identified by file name:

| Figure file | Content | Drawn by |
|---|---|---|
| `Fig_pinn_train.svg` | (a) validation and training error versus epoch for 10 %, 50 % and 100 % of the training data; (b) predicted versus simulated displacement for held-out designs, with R² and MAE | `figA.py` |
| `Fig_field.svg` | (a) membrane deflection shape (the clamped-edge shape function, scaled to the largest displacement in the database); (b) waveguide field Im(Ez) | `figB.py` |
| `Fig_grad.svg` | (a) exact adjoint gradient map; (b) adjoint versus finite-difference gradients, with cosine similarity and largest relative error | `figB.py` |
| `Fig_inv.svg` | (a) best objective versus number of surrogate queries (probes + gradient versus random search); (b) objective landscape over `t_e` and `t_np` with the optimum | `figC1.py` |
| `Fig_yield.svg` | (a) distributions of performance under fabrication errors for the nominal, mean-variance-corrected and (after `cvar_study.py`) CVaR-corrected designs; (b) improvement of the robustness score during the correction search | `figC1.py` (+ `cvar_study.py`) |
| `Fig_transfer.svg` | Nominal, transferred-correction and re-optimised scores at 3.0 to 6.0 MHz | `figC2.py` |
| `Fig_ablation.svg` | (a) inverse-design components at a 360-query budget; (b) nominal, mean-variance and CVaR designs on mean, 5th percentile and CVaR | `fig_ablation.py` |

The previous version of this README also mentioned three hand-drawn diagrams
(`abstract_Fig.svg`, `Fig_1.svg`, `Fig_4.svg`); they are not in this
repository.

---

## 7. The Python modules

| File | What it contains |
|---|---|
| `parl_id/physics/cmut_plate.py` | Plate equation of the membrane in frequency domain, written as two second-order equations (deflection and moment) so the network needs only second derivatives; electrostatic pull of the gap; layer-stack stiffness and mass |
| `parl_id/physics/helmholtz.py` | Wave-equation (Helmholtz) residual for the waveguide PINN; a small sparse finite-difference solver with absorbing edges; its exact adjoint gradient |
| `parl_id/physics/acoustic_loading.py` | Radiation impedance of a flat piston in water (loading by the surrounding fluid); tested, but not used in the PINN training loss |
| `parl_id/pinn/model.py` | Network with random Fourier input features; a wrapper that makes the deflection exactly zero, with zero slope, on the edge of the unit square |
| `parl_id/pinn/losses.py` | Combined loss (data + equation + boundary) with automatic weighting; optional ranking (Pearson correlation) term, off by default (`w_corr=0`) |
| `parl_id/pinn/train.py` | Builds the CMUT PINN (inputs x, y, four thicknesses, frequency; outputs deflection and moment) and trains it |
| `parl_id/adjoint/neural_adjoint.py` | Box-limited design vector; optimiser (Adam or L-BFGS, several random restarts) through a frozen network; gradient cosine similarity; `SpecWarmStartBank` (reuse the design solved for the nearest target; not called by any script) |
| `parl_id/adjoint/projections.py` | Round thicknesses to a manufacturable step; erode/dilate a 2-D pattern (not called by any script) |
| `parl_id/rl/fab_env.py` | Simulated fabrication loop in the `gymnasium` format: random process errors, CVaR or mean-minus-spread score, bounded corrections |
| `parl_id/rl/gcn_sac.py` | Main agent: the design is a small graph (one node per thickness plus a target node); a graph neural network feeds a soft actor-critic (SAC) agent that learns from a prioritized replay memory (past steps with a large prediction error are replayed more often) |
| `parl_id/rl/ppo_agent.py` | A minimal PPO agent (comparison baseline) |
| `parl_id/rl/evo_warmstart.py` | A short evolutionary search whose best correction is used as the agent's starting point |
| `parl_id/data/cmut_loader.py` | Reads the database CSV; design ranges; synthetic stand-in database |
| `parl_id/data/ceviche_loader.py` | Small waveguide dataset from the built-in solver; optional wrapper for `ceviche-challenges` |
| `experiments/make_figures/common.py` | Figure style (Okabe-Ito colour-blind-safe colours), input/output paths, per-design objective (peak or band-average displacement within ±0.5 MHz of the target) |

---

## 8. Where the numbers come from

**Database.** The finite-element results are on Zenodo
(https://doi.org/10.5281/zenodo.21290617). The loader expects columns named
`t_e, t_np, t_ox, t_w, frequency_MHz, displacement_um` (read by name, so the
order does not matter); a new export with these columns can be passed with
`--csv`. The previous README asks users to evaluate on **held-out design
combinations**, not held-out rows (it points to "paper, Sec. VI-A").

**Design ranges** (`BOUNDS` in `parl_id/data/cmut_loader.py`, noted in the
code as taken from the previous paper), in micrometres: `t_e` 0.44 to 0.84,
`t_np` 1.3 to 2.5, `t_ox` 1.3 to 2.5, `t_w` 2.9 to 3.5. These ranges are used
by the PyTorch stages. The figure scripts instead use the smallest and largest
values found in the database file.

**Plate and material values** (defaults in `CMUTPlateResidual`; no source
is recorded in the code): nitride Young's modulus 250 GPa, Poisson ratio 0.23,
density 3100 kg/m³; gold Young's modulus 79 GPa, density 19300 kg/m³; gap
100 nm; voltage 5 V. The training loop uses a 33 µm length scale and a 10 MHz
reference frequency.

**Fabrication errors** (defaults in `FabricationEnv`, repeated in the figure
scripts). The code says these follow published CMUT fabrication statistics but
does not name the source:

| Error | Model |
|---|---|
| Film thickness (`t_e`, `t_np`) | multiplied by a random factor, mean 1, spread 3 % |
| Wall over/under-etch (`t_w`) | plus a random amount, spread 0.08 µm |
| Stress (applied to `t_ox`) | multiplied by a random factor, mean 1, spread 5 % |

Other defaults of the environment: 16 random fabrication draws per score,
CVaR with `alpha = 0.05` (with 16 draws this is the single worst draw, see
Section 10), correction per step at most 15 % of each range,
8 steps per episode.

**Waveguide test problem**: wavelength 1.55 µm, silicon permittivity 12.25,
40 nm grid, 10 absorbing cells at each edge.

**Acoustic loading** defaults: water, density 1000 kg/m³, sound speed
1500 m/s.

**Results recorded by the repository** (from the previous README and the
labels typed into the figure scripts; not re-computed for this guide because
the database could not be downloaded here): best design
θ* = (0.352, 0.972, 0.966, 3.018) µm with J* = 0.0813 µm at 4.3 MHz; random
search needs 4,817 queries to match the 360-query result (13.4 times fewer
queries); held-out R² = 0.86; the CVaR-corrected design improves the mean by
5.5 %, the 5th percentile by 5.2 % and CVaR by 5.0 %; the specification
warm start keeps 95 to 102 % of the cold-start result at 11 % of the query
budget; reusing a fixed correction at other frequencies recovers −65 % of the
attainable gain.

---

## 9. Built-in checks

**`tests/test_smoke.py`** (eight tests; they check that each part runs and
gives finite, correctly shaped output, not that results are accurate):

- `test_pinn_forward`: output shape, and zero deflection on the edge `x = 0`.
- `test_plate_residual`: the plate-equation residual has the right shape and
  finite values.
- `test_train_few_epochs`: 3 training epochs on a tiny synthetic database.
- `test_helmholtz_fdfd_and_residual`: the finite-difference solver gives a
  finite, non-zero field; the Helmholtz residual is finite.
- `test_neural_adjoint_quadratic`: the design optimiser finds the minimum of a
  simple quadratic (objective below 1e-3).
- `test_fab_env_and_ppo`: the fabrication environment and one PPO iteration.
- `test_acoustic_loading`: the radiation impedance is finite.
- `test_gcn_sac_per`: 60 GCN-SAC steps, and the replay memory updates its
  priorities.

**Gradient check in `figB.py`** (does not use the database): the exact
adjoint gradient is compared with central finite differences at 40 random
pixels. In our run it printed cosine similarity 1.0000000000000002 and
largest relative error 6.6e-6.

**`extensions_study.py`, part 3**: 5-fold check of how well the interpolated
objective ranks held-out designs (Pearson and Spearman correlation).

---

## 10. Notes on the calculations

- **Units.** Thicknesses in micrometres, frequency in MHz, displacement in
  micrometres. Inside the PyTorch code the frequency is divided by 10 MHz and
  the displacement by 0.01 µm.
- **What the PINN learns from data.** The data term compares the average of
  the predicted deflection over random points on the membrane with the
  database value for that design and frequency.
- **Two different surrogates.** The PyTorch stages use the PINN. The figure
  scripts do not use PyTorch: `figA.py` trains a small numpy network (two
  hidden layers of 48 units, 25 epochs), and `figC1.py`, `cvar_study.py`,
  `figC2.py` and `extensions_study.py` interpolate the per-design objective
  with thin-plate-spline radial basis functions and take finite-difference
  gradients. The corrections in `Fig_yield` and `Fig_transfer` come from a
  random local search on that interpolated objective, not from the RL agent.
- **Numbers typed into the figure scripts.** `fig_ablation.py` draws values
  written directly in the script. Some annotation boxes in `figC1.py`
  ("4,817 queries", "13.4×", "+5.2 %", "+5.5 %, +5.2 %, +5.0 %") and
  `figC2.py` ("−65 %") are also fixed text. The values the scripts actually
  compute are printed on screen; compare them with the labels if you change
  the data.
- **Repeatability.** The figure scripts and the RL environment use fixed
  random seeds. The PyTorch stages do not set a PyTorch seed (network
  weights, training batches and the random points used to average the
  deflection are drawn freshly), so repeated runs differ slightly.
- **Gradient check in `run_photonic_pinn.py`.** Its final finite-difference
  check uses a different source than the reference field it compares with. In
  our run it printed relative errors of 1.0 at all five pixels. Use the check
  in `figB.py` (Section 9), which uses the same source throughout.
- **CVaR with few draws.** The environment keeps the worst
  `k = max(1, ceil(alpha x K))` of its `K` draws. With the defaults
  (`alpha = 0.05`, `K = 16`) that is `k = 1`, so the RL score is the single
  worst draw, not a smooth 5 % tail average. `cvar_study.py` uses
  `k = max(1, int(alpha x K))` with `K = 200` (search, 10 draws kept) and
  `K = 2000` (comparison, 100 draws kept), a true 5 % tail. `figC1.py` and
  `figC2.py` score robustness as mean minus one standard deviation.
- **Out-of-date docstring.** The docstring at the top of
  `run_rl_fab_loop.py` says "PPO learns a yield-aware correction", but the
  default agent is GCN-SAC (`--agent gcn-sac`); PPO is used only with
  `--agent ppo`.
- **Waveguide solver.** With the source convention used here the field in a
  lossless region is purely imaginary, which is why `Fig_field` shows
  Im(Ez). The absorbing edge cells are cropped from the plots.

### Methods this code builds on

These are cited in the code docstrings and the earlier README exactly as
below; this guide adds no further bibliographic details.

- **CVaR** tail-risk score: "Rockafellar–Uryasev" (earlier README), default
  score in `parl_id/rl/fab_env.py`.
- **Graph convolutional encoder**: "Kipf–Welling" (earlier README),
  `parl_id/rl/gcn_sac.py`.
- **Soft actor-critic**: "Haarnoja" (earlier README), `parl_id/rl/gcn_sac.py`.
- **Prioritized experience replay**: "Schaul et al., 2016"
  (`parl_id/rl/gcn_sac.py`).
- **Evolutionary warm start**: inspired by "Evo-PHORCED (Meghwar et al.,
  2025)" (`parl_id/rl/evo_warmstart.py`).
- **Rank-aware (Pearson correlation) loss**: inspired by "PearSAN, Bezick et
  al., Adv. Optical Materials 2026" (`parl_id/pinn/losses.py`).
- **Specification warm-start bank**: inspired by "Soda-PTA ... (Sun et al.,
  2025)" (`parl_id/adjoint/neural_adjoint.py`, `extensions_study.py`).
- **Bandwidth-aware objective**: "industrial practice, cf. TSMC dual-layer
  grating couplers" (`combo_objective_band` in
  `experiments/make_figures/common.py`).
- **Random Fourier features**: "Tancik et al." (`parl_id/pinn/model.py`).

---

## 11. Version history

| Version | Date | Notes |
|---|---|---|
| Unreleased | 30 Sep 2026 | Fixes: figure SVGs and the PowerPoint deck now written inside the repository (`outputs/`); PyTorch minimum raised to 2.3 |
| Unreleased | 30 Sep 2026 | Documentation only: README rewritten, CHANGELOG added, CITATION.cff corrected |
| Initial code (no version tag; `parl_id.__version__` is `0.1.0`) | 25 Jul 2026 | First public code |

The repository has no tagged releases. Details are in
[CHANGELOG.md](CHANGELOG.md).

---

## 12. How to cite

GitHub shows a **"Cite this repository"** button in the right-hand column,
which reads `CITATION.cff`. Please cite the software, the earlier paper and
the database:

> T. M. Mahim, M. M. Rahman, and A.H.M.A. Rahim, "PARL-ID: Fabrication-Aware
> Physics-Informed Adjoint Framework With Reinforcement Learning in the Loop
> for Inverse Design of CMUT and Photonic Sensors," software,
> https://github.com/Tanvir-Mahmud-Mahim/Fabrication-Aware-Physics-Informed-Adjoint-Framework-With-RL-in-the-Loop-for-Inverse-Design-of-CMUT
>
> T. M. Mahim, M. M. Rahman, and A.H.M.A. Rahim, "Hierarchical Inverse Design
> Framework for Unit-Cell CMUTs With Attentive Gated Recurrent and Fully
> Connected Dense Layers," IEEE Sensors Journal 25(13), 26104-26116 (2025),
> https://doi.org/10.1109/JSEN.2025.3569424
>
> Open FEM Benchmark Database for CMUT Inverse Design, Zenodo,
> https://doi.org/10.5281/zenodo.21290617

---

## 13. License and contact

Code: Apache License 2.0 (see `LICENSE`).

Questions and bug reports: please open an issue on this repository, or
contact Tanvir M. Mahim, BRAC University (tanvir.mahim@bracu.ac.bd).
