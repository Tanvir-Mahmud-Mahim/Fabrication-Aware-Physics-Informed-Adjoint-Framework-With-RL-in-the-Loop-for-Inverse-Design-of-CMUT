# PARL-ID — Physics-Informed Adjoint Inverse Design with RL in the Fabrication Loop

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-orange.svg)](https://doi.org/10.5281/zenodo.21290617)

Python codebase for the paper *"PARL-ID: A Fabrication-Aware Physics-Informed
Adjoint Framework With Reinforcement Learning in the Loop for Inverse Design of
CMUT and Photonic Sensors"* (IEEE Sensors Journal, in preparation), building on
our previous work [doi:10.1109/JSEN.2025.3569424](https://doi.org/10.1109/JSEN.2025.3569424).

**Highlights**: multi-physics PINN surrogate (Kirchhoff–Love plate + Helmholtz)
· neural-adjoint inverse engine validated to the truncation-error floor ·
graph-encoded soft actor-critic with prioritized replay optimizing the
**CVaR tail-risk yield** of the fabricated-performance distribution · open
55,350-sample CMUT benchmark database
([doi:10.5281/zenodo.21290617](https://doi.org/10.5281/zenodo.21290617)).

## Setup (VSCode, Windows/Linux)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux)
pip install -r requirements.txt
pip install ceviche-challenges  # optional: exact photonic benchmark
```

## Layout

```
parl_id/
├── physics/           PDE residuals & exact solvers
│   ├── cmut_plate.py       Kirchhoff–Love plate + electrostatic coupling (mixed form)
│   ├── helmholtz.py        2D Helmholtz residual + minimal sparse FDFD + exact adjoint
│   └── acoustic_loading.py Reduced-order radiation-impedance fluid loading
├── pinn/              Fourier-feature PINN, hard-BC wrapper, composite loss, training
├── adjoint/           Neural-adjoint optimizer, fabrication projections
├── rl/                Gymnasium fabrication env (process corruption) + minimal PPO
├── data/              CMUT FEM database loader (+ synthetic fallback), ceviche wrapper
experiments/
├── augmented_reshaped_dataset.csv   the released FEM database (369 combos × 150 freq. points)
├── Final_dataset.csv                pre-augmentation export
├── run_*.py                         one script per PyTorch experiment
└── make_figures/                    generates every result SVG of the paper (numpy/scipy only)
tests/                 Smoke tests
```

## Reproducing the paper pipeline (PyTorch experiments)

```bash
# 1. Train the CMUT multi-physics PINN (Fig. F2/F3)
python experiments/run_cmut_pinn.py --csv experiments/augmented_reshaped_dataset.csv
#    (omit --csv to use the physics-flavored synthetic database)

# 2. Neural-adjoint inverse design (Fig. F5)
python experiments/run_adjoint_design.py --target-freq 4.3

# 3. RL fabrication loop: yield-aware robust correction (Fig. F7/F8)
python experiments/run_rl_fab_loop.py

# 4. Photonic testbench + gradient-fidelity validation (Fig. F4/F6)
python experiments/run_photonic_pinn.py

# smoke tests
python tests/test_smoke.py
```


## Regenerating the paper figures (no PyTorch required)

The result figures in `latex/` are produced from the real FEM database by
numpy/scipy-only scripts (`pip install numpy scipy pandas matplotlib`):

```bash
cd experiments/make_figures
python figA.py              # Fig_pinn_train.svg — surrogate convergence + held-out parity (R² = 0.86)
python figB.py              # Fig_field.svg, Fig_grad.svg — FDFD field + exact-adjoint validation
python figC1.py             # Fig_inv.svg, Fig_yield.svg — seeded-gradient inverse design + yield CDFs
python cvar_study.py        # CVaR tail-risk correction (paper Eq. 17, FOM table panel C)
python figC1.py             # rerun: picks up cvar_state.npz -> third CDF curve in Fig_yield
python figC2.py             # Fig_transfer.svg — correction-transfer study
python extensions_study.py  # spec warm-start, rank quality, bandwidth objective
python fig_ablation.py      # Fig_ablation.svg — component ablation (Sec. VI-F)
```

SVGs are written directly into `../../latex/`. `figC1.py` stores its optimum in
`state_c1.npz` (consumed by `cvar_study.py`, `figC2.py`, `extensions_study.py`)
— keep the order above. Key computed numbers: θ* = (0.352, 0.972, 0.966, 3.018) µm,
J* = 0.0813 µm at 4.3 MHz (13.4× query efficiency vs. random search);
CVaR-corrected design dominates every yield FOM (mean +5.5%, P5 +5.2%,
CVaR5 +5.0%); spec warm-start retains 95–102% of cold-start quality at 11%
budget; static-correction transfer recovery −65% (motivates the RL policy).

## The RL agent: graph-encoded SAC with prioritized replay

`parl_id/rl/gcn_sac.py` implements the main Stage-3 agent (paper Sec. V-C):

- **Device graph**: one node per design parameter + a specification node
  (target frequency), edges = physical adjacency in the CMUT layer stack.
- **GCN encoder** (Kipf–Welling): mean-pooled embedding whose size is
  independent of the parameter count → the policy architecture is portable
  across device families (CMUT thickness vector ↔ photonic density patch).
- **Soft actor-critic** (Haarnoja): off-policy — every expensive
  corruption-ensemble evaluation is stored and reused; twin Q-critics,
  automatic entropy tuning.
- **Prioritized experience replay** (Schaul): P(i) ∝ (|TD error|+ε)^ω with
  annealed importance weights — under the CVaR reward this concentrates
  learning on the rare lower-tail (wafer-killing) outcomes.

```bash
python experiments/run_rl_fab_loop.py                     # GCN-SAC + PER + CVaR (default)
python experiments/run_rl_fab_loop.py --agent ppo         # PPO baseline (ablation)
python experiments/run_rl_fab_loop.py --reward mean_std   # mu - beta*sigma reward (ablation)
python experiments/run_rl_fab_loop.py --no-evo-warmstart  # cold-start ablation
```

## Mechanisms adopted from contemporary SOTA (2025–2026)

- **CVaR tail-risk yield reward** (novel here; Rockafellar–Uryasev): default in
  `parl_id/rl/fab_env.py` (`reward_mode="cvar"`, `alpha=0.05`); `mean_std` is
  the ablation baseline.
- **Evolutionary policy warm start** (Evo-PHORCED-inspired, applied to the
  yield reward): `parl_id/rl/evo_warmstart.py`; enabled by default.
- **Rank-aware surrogate loss** (PearSAN-inspired, combined with physics):
  `pearson_correlation_loss` in `parl_id/pinn/losses.py`; activate with
  `CompositeLoss(w_corr=...)` and pass `pred_target=(pred, target)`.
- **Specification warm-start bank** (Soda-PTA-inspired, transfers designs, not
  hyperparameters): `SpecWarmStartBank` in `parl_id/adjoint/neural_adjoint.py`.
- **Bandwidth-aware objective** (industrial practice): `combo_objective_band`
  in `experiments/make_figures/common.py`.

Figure styling is centralized in `experiments/make_figures/common.py`
(Okabe-Ito colorblind-safe palette, annotation helpers); every figure script
regenerates its polished SVG directly into `latex/`.

The three diagram figures (`abstract_Fig.svg`, `Fig_1.svg`, `Fig_4.svg`) are
hand-drawn editable SVGs in `latex/` (edit in Inkscape/Illustrator). Editable
native-shape versions of all figures live in `../PARL-ID_figures.pptx`,
rebuilt by `experiments/make_figures/deck.js` (`npm install pptxgenjs`,
`node deck.js`).

## Manuscript, supplementary and dataset

- `latex/main.tex` — main article (references auto-number 1…N in citation
  order; abstract ≤250 words). `latex/supplementary.tex` — Supplementary
  Material (device geometry, forward/gradient validation figures, SOTA
  comparison table, adopted-mechanism quantification, hyperparameters).
- Dataset release package: `../dataset_release/` (benchmark CSV + README +
  `UPLOAD_GUIDE.md` with the 5-minute Zenodo steps). After Zenodo mints the
  DOI, paste it into `\newcommand{\datasetdoi}{...}` in BOTH tex files —
  every dataset link updates automatically.

## Compiling the LaTeX manuscript

`latex/main.tex` compiles with XeLaTeX/pdfLaTeX + `-shell-escape` (Inkscape on
PATH for `\includesvg`). The preamble sets `\svgsetup{inkscapelatex=false}` so
SVG text is rendered by Inkscape rather than extracted into LaTeX — this is
required because figure labels such as `t_np` would otherwise trigger
`Missing $ inserted` errors. **If you compiled before this fix, delete the
generated `latex/svg-inkscape/` cache folder once before recompiling.**

## Dropping in a new FEM export

Export the COMSOL parametric sweep as CSV with columns
`t_e, t_np, t_ox, t_w, displacement_um, frequency_MHz` and pass it via
`--csv`. Release the same CSV on Zenodo to update the open CMUT benchmark
cited in the paper.

## External benchmarks

- **ceviche-challenges** (Google): `pip install ceviche-challenges` — used via
  `parl_id/data/ceviche_loader.py` for exact FDFD forward/adjoint on standard
  problems (mode converter, beam splitter, waveguide bend, WDM).
- **invrs-gym** (invrs.io): JAX-based challenges + public solution leaderboard,
  used as an external comparison set.
- **MetaNet** (Stanford): >100k metagrating designs, optional pretraining data.

Hardware: everything runs on CPU; a 4 GB GPU (GTX 1650) accelerates PINN
training. PINNs here are small MLPs (~200k parameters).

## Dataset

The FEM benchmark database (369 design combinations × 150 frequency points)
is hosted on Zenodo: [doi:10.5281/zenodo.21290617](https://doi.org/10.5281/zenodo.21290617).
Download `cmut_inverse_design_benchmark.csv`, place it at
`experiments/augmented_reshaped_dataset.csv`, and every script runs as-is.
Please evaluate with **held-out design combinations**, not rows (see paper,
Sec. VI-A).

## License and citation

Licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)).
If you use this code or the dataset, please cite the papers listed in
[CITATION.cff](CITATION.cff).
