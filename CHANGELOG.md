# Changelog

All notable changes to this code are listed here, newest first.
The repository has no tagged releases yet.

## Unreleased: documentation (30 September 2026)

Documentation only; no code, data or results changed.

- README rewritten as a step-by-step guide: the idea in plain language,
  file tree, installation, three ways to use the code, tables of scripts
  (with run times measured on a shared two-core computer), figure files,
  modules, parameter sources, built-in checks and notes on the calculations.
- Removed README references to files that are not in this repository:
  `experiments/augmented_reshaped_dataset.csv` and `Final_dataset.csv`
  (the database is on Zenodo, https://doi.org/10.5281/zenodo.21290617),
  `latex/main.tex`, `latex/supplementary.tex`, the LaTeX build notes,
  `../dataset_release/` with `UPLOAD_GUIDE.md`, the instruction to paste the
  dataset DOI into the LaTeX files, `../PARL-ID_figures.pptx`, and the
  hand-drawn diagram SVGs.
- Removed the "External benchmarks" mentions of invrs-gym and MetaNet, which
  no code in this repository uses.
- Kept the credits to the methods the code builds on (CVaR, GCN, SAC,
  prioritized replay, Evo-PHORCED, PearSAN, Soda-PTA, the TSMC-inspired band
  objective, Fourier features), cited exactly as in the code and the earlier
  README, and the code's own figure/equation/section labels
  ("Fig. F2/F3" ... "Eq. 16/17").
- Stated that with the environment's default 16 draws the CVaR score is the
  single worst draw, and that the `run_rl_fab_loop.py` docstring names PPO
  although the default agent is GCN-SAC.
- Run times now give the exact arguments and note that they varied several-fold
  with machine load.
- Corrected the figure output folder: the scripts write SVGs to a folder
  named `latex` next to the repository folder, not inside it.
- Corrected the PINN size: the CMUT and waveguide networks each have 82,818
  trainable parameters (the old README said about 200,000).
- Added this CHANGELOG.
- `CITATION.cff`: removed the dataset entry from `references`. It had no
  authors, which made the file invalid under the Citation File Format 1.2.0
  schema, and the dataset authors could not be verified from the repository.
  The dataset DOI (https://doi.org/10.5281/zenodo.21290617) stays in the
  README.

## Initial code (25 July 2026)

First public code, added in a series of commits (no version tag; the
package reports version `0.1.0` in `parl_id/__init__.py`).

- README, Apache-2.0 LICENSE, CITATION.cff, requirements.txt, .gitignore.
- `parl_id/physics`: plate electro-mechanics, Helmholtz/finite-difference
  solver with exact adjoint, acoustic loading.
- `parl_id/pinn`: Fourier-feature network, hard boundary wrapper, losses,
  training.
- `parl_id/adjoint`: neural-adjoint optimiser, specification warm-start bank,
  fabrication projections.
- `parl_id/rl`: CVaR fabrication environment, GCN-SAC with prioritized
  replay, PPO baseline, evolutionary warm start.
- `parl_id/data`: CMUT database loader, ceviche wrapper.
- `experiments/`: PINN training, adjoint design, RL fabrication loop,
  photonic testbench.
- `experiments/make_figures/`: figure scripts and the PowerPoint deck builder.
- `tests/test_smoke.py`: smoke tests.
