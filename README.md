# PARL-ID: Physics-Informed Adjoint Inverse Design with RL in the Fabrication Loop

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21290617.svg)](https://doi.org/10.5281/zenodo.21290617)

This repository holds the Python code for our manuscript:

**"PARL-ID: A Fabrication-Aware Physics-Informed Adjoint Framework With Reinforcement Learning in the Loop for
Inverse Design of CMUT and Photonic Sensors"**

I wrote it with M. Mosaddequr Rahman and A.H.M.A. Rahim (BRAC University).
The manuscript is in preparation for IEEE Sensors Journal.

It builds on our earlier paper in IEEE Sensors Journal 25(13),
26104-26116 (2025), https://doi.org/10.1109/JSEN.2025.3569424.

- Repository: https://github.com/Tanvir-Mahmud-Mahim/Fabrication-Aware-Physics-Informed-Adjoint-Framework-With-RL-in-the-Loop-for-Inverse-Design-of-CMUT
- CMUT simulation database (369 designs x 150 frequencies = 55,350 rows): https://doi.org/10.5281/zenodo.21290617

You will not find the manuscript itself (LaTeX sources) here.

---

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [Installation](#3-installation)
4. [Quick start: three ways to use the code](#4-quick-start-three-ways-to-use-the-code)
5. [The scripts, step by step](DETAILS.md#5-the-scripts-step-by-step)
6. [Which script makes which figure](DETAILS.md#6-which-script-makes-which-figure)
7. [The Python modules](DETAILS.md#7-the-python-modules)
8. [Where the numbers come from](DETAILS.md#8-where-the-numbers-come-from)
9. [Built-in checks](DETAILS.md#9-built-in-checks)
10. [Notes on the calculations](DETAILS.md#10-notes-on-the-calculations)
11. [Version history](DETAILS.md#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

Sections 5 to 11 are in [DETAILS.md](DETAILS.md), with a short guide in
[More details](#more-details-sections-5-to-11). DETAILS.md also has
[extra notes for Sections 1 to 4](DETAILS.md#extra-notes-for-sections-1-to-4).

---

## 1. The idea in one minute

A **CMUT** (capacitive micromachined ultrasonic transducer) is a tiny drum.
It is a thin membrane held over a narrow gap. A voltage moves it to send and
receive ultrasound. How strongly it moves at a given frequency depends on the
thicknesses of its layers. Here the design has four numbers, all in
micrometers: the electrode thickness `t_e`, the nitride layer thickness
`t_np`, the oxide thickness `t_ox`, and the support-wall thickness `t_w`.

The goal is **inverse design**. That means choosing those four numbers so that the membrane
moves as much as possible at a target frequency (4.3 MHz by default). The design
should also stay good when the real fabrication process makes every layer
slightly wrong. The code does this in three stages:

1. **A fast stand-in model (surrogate).** A neural network is trained on
   finite-element simulation results. In the PyTorch code it is a
   **physics-informed neural network (PINN)**. Its training penalizes
   mismatch with the data, and it also penalizes breaking the plate equation of the membrane.
2. **Gradient-based design (the "adjoint" step).** The surrogate is
   differentiable. So the code can compute how the objective changes with each
   design number and walk uphill. ("Adjoint" is the standard name for this
   efficient way of getting gradients.)
3. **Fabrication loop with reinforcement learning (RL).** A simulated
   fabrication step randomly perturbs the design (film thickness, etching,
   stress). An RL agent learns a small correction that improves the
   **worst-case** outcomes. The default score is **CVaR** (conditional
   value-at-risk). It is the average of the worst 5 % of the perturbed results,
   and never fewer than one result.

The code also uses a second, simpler problem to check the gradient machinery
against an exact answer. It is light traveling in a silicon waveguide, solved
with a small finite-difference solver.

**Main results.** These come from the previous README and the
labels typed into the figure scripts. I did not re-compute them for this guide, because
I could not download the database here.

- The best design is θ* = (0.352, 0.972, 0.966, 3.018) µm with J* = 0.0813 µm at 4.3 MHz.
- Random search needs 4,817 queries to match the 360-query result (13.4 times fewer queries).
- The held-out R² is 0.86.
- The CVaR-corrected design improves the mean by 5.5 %, the 5th percentile by 5.2 % and CVaR by 5.0 %.

All recorded results are in
[Section 8](DETAILS.md#8-where-the-numbers-come-from). How CVaR works with the
default number of draws is in
[DETAILS.md](DETAILS.md#more-on-section-1-the-idea-in-one-minute).

---

## 2. What is in this repository

The main parts are:

- **parl_id/**: the Python package (physics, PINN, gradient-based design, RL, data loaders).
- **experiments/**: the three PyTorch stages and the waveguide test problem.
- **experiments/make_figures/**: numpy/scipy scripts that draw the result figures (no PyTorch).
- **tests/test_smoke.py**: eight quick checks that every part runs.
- **requirements.txt**: Python packages to install.
- **CHANGELOG.md**, **CITATION.cff** and **LICENSE**.

The full file tree, with a note on each file, is in
[DETAILS.md](DETAILS.md#more-on-section-2-the-full-file-tree).

**The database is not stored in this repository.** You can download it from Zenodo
(see [Way B](#way-b-redraw-the-result-figures-from-the-database)).
The scripts write results to `outputs/` (PyTorch stages; the figure scripts
write their SVG files to `outputs/figures/`). They also write small state files (`.npy`/`.npz`) to
`experiments/make_figures/`. Git ignores all of these.
No script writes outside the repository.

---

## 3. Installation

The repository does not state a minimum Python version. I used
**Python 3.11** for the checks in this guide.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

`requirements.txt` installs `torch` (2.3 or newer; 2.4.1 or newer on
Windows), `numpy`, `scipy`, `pandas`, `matplotlib`, and `gymnasium` (the
standard interface for RL environments). You need PyTorch only for the PINN/adjoint/RL stages and the
tests. The figure scripts need only `numpy`, `scipy`, `pandas`, and
`matplotlib`. If you want a smaller download without GPU support, pick the
CPU build on the PyTorch website before running the line above.

The scripts use a CUDA GPU automatically when PyTorch finds one (the RL loop
always runs on the CPU). Everything also runs on a CPU.

Optional extras, the reason for PyTorch 2.3, and the minimum versions I
checked are in [DETAILS.md](DETAILS.md#more-on-section-3-installation).

---

## 4. Quick start: three ways to use the code

Run all commands from the repository folder unless a step says otherwise.

### Way A: check that everything works (under half a minute)

```
python tests/test_smoke.py
```

Each of the eight tests prints `PASS ...`, and the last line reads
`all smoke tests passed`. You may see a PyTorch `UserWarning` about converting a tensor
to a number. It is harmless.

### Way B: redraw the result figures from the database

You do not need PyTorch for this.

1. Download `cmut_inverse_design_benchmark.csv` from
   https://doi.org/10.5281/zenodo.21290617 and save it as
   `experiments/augmented_reshaped_dataset.csv`.
2. Run the scripts in this order, because later ones read files written by earlier
   ones:

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

   `figC1.py` runs twice. The second run picks up `cvar_state.npz` from
   `cvar_study.py` and adds the third (CVaR) curve to `Fig_yield.svg`.

### Way C: run the PyTorch pipeline

```
python experiments/run_cmut_pinn.py --csv experiments/augmented_reshaped_dataset.csv
python experiments/run_adjoint_design.py --target-freq 4.3
python experiments/run_rl_fab_loop.py
```

If you leave out `--csv`, the first script trains on a built-in synthetic stand-in database instead
of the real one. The second script needs `outputs/cmut_pinn.pt` from the
first. The third needs both `outputs/cmut_pinn.pt` and
`outputs/adjoint_design.pt`. The first and third take a long time at their default
settings. For a quick trial run, add `--epochs 5` and `--rl-iters 1`.

---

## More details (Sections 5 to 11)

The full notes are in [DETAILS.md](DETAILS.md). Here is what each section holds:

- [5. The scripts, step by step](DETAILS.md#5-the-scripts-step-by-step):
  every script with what it does, rough run times and output files; the options of `run_rl_fab_loop.py`; `deck.js`.
- [6. Which script makes which figure](DETAILS.md#6-which-script-makes-which-figure):
  the figure labels used in the code, and what each SVG figure file shows.
- [7. The Python modules](DETAILS.md#7-the-python-modules):
  one line on each file of the Python package.
- [8. Where the numbers come from](DETAILS.md#8-where-the-numbers-come-from):
  the database columns, design ranges, material values, fabrication errors and the recorded results.
- [9. Built-in checks](DETAILS.md#9-built-in-checks):
  what the eight smoke tests check, and the gradient check in `figB.py`.
- [10. Notes on the calculations](DETAILS.md#10-notes-on-the-calculations):
  units, the two surrogates, fixed numbers in the figure scripts, repeatability, known caveats, and the methods the code builds on.
- [11. Version history](DETAILS.md#11-version-history):
  each version with its date, and a link to CHANGELOG.md.

---

## 12. How to cite

GitHub shows a **"Cite this repository"** button in the right-hand column.
It reads `CITATION.cff`. Please cite the software, our earlier paper and
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

If you have questions or find a bug, please open an issue on this repository. You can also
contact me, Tanvir M. Mahim, at BRAC University (tanvir.mahim@bracu.ac.bd).
