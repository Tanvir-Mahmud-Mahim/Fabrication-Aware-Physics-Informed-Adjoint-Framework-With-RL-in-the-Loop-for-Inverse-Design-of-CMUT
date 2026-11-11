# PARL-ID: Physics-informed Adjoint and Robust-Loop Inverse Design (CMUTs)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-orange.svg)](https://doi.org/10.5281/zenodo.21290617)

Code for the article *"PARL-ID: Physics-Informed and Yield-Aware Inverse Design of
Capacitive Micromachined Ultrasonic Transducers With Neural-Adjoint Optimization"*
(revised manuscript, under review), building on
[doi:10.1109/JSEN.2025.3569424](https://doi.org/10.1109/JSEN.2025.3569424).

**All results of the revised article and its supplementary material are produced by
`experiments/revision/` (NumPy, SciPy, autograd, scikit-learn, ceviche) and by
`experiments/make_figures/figB.py`.**

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install numpy scipy pandas matplotlib autograd scikit-learn ceviche ceviche-challenges cairosvg
```

Download `cmut_inverse_design_benchmark.csv` from Zenodo
([doi:10.5281/zenodo.21290617](https://doi.org/10.5281/zenodo.21290617)) and save it as
`experiments/augmented_reshaped_dataset.csv`.

## Important note on the database

The file has 369 thickness combinations, but **only 123 are FEM designs**; the other
246 are augmentation copies (inputs perturbed up to 12 %, outputs copied). The scripts
use only the 123 FEM designs and hold out whole designs for evaluation. The
t_e = 0.24 um block shows a pattern that suggests an input/output misalignment; see
`e0_data_audit.py` and Sec. IV-B of the article.

## Reproducing the revised article (`experiments/revision/`)

Each script is resumable (results cached in `results/`); the time limits are in seconds.

| Script | Content | Article |
|---|---|---|
| `e0_data_audit.py` | database audit, GP reference, original-pipeline check | Sec. IV-B, VI-C; Table S-III |
| `e1_surrogate.py 40 0 1` | physics-informed vs data-only surrogate, ablations (5-fold, held-out design) | Sec. VI-A, Fig. 3, Table S-IV |
| `e2_design.py` | surrogate ensemble, corruption model, inverse engines, robust optimizer (library) | Sec. V |
| `e2b_engine.py`, `e2c_max.py` | inverse-engine comparisons | Sec. VI-C, Fig. 4 |
| `e2d_robust.py`, `e2e_noise.py` | robust correction, estimator noise | Sec. VI-D, Fig. 5, Tables S-V, S-VI |
| `e3_rl.py <agent> <seed> <s>` | GCN-SAC (+/-PER), MLP-SAC, GCN-PPO | Sec. VI-E, Fig. 6 |
| `e3b_policy.py <seed> <s>`, `e3c_eval.py` | adjoint-trained policy, zero-shot evaluation | Sec. VI-E, Table S-VII |
| `e4_photonic.py data/grad/fit/opt/yield` | ceviche waveguide bend: gradient fidelity, robust design | Sec. VI-F, Fig. 7, Table S-VIII |
| `make_revision_figures.py` | Figs. 3-7 (SVG into `../../../latex/`) | |

Agents for `e3_rl.py`: `sac_gcn_per`, `sac_gcn_uniform`, `sac_mlp_per`, `ppo_gcn`.
`make_figures/figB.py` produces Supplementary Figs. S2 and S3 (FDFD field and adjoint check).

## Other folders

* `parl_id/` and `experiments/run_*.py`: PyTorch modules of the original formulation
  (field PINN, neural adjoint, GCN-SAC/PPO). They were not used for the revised results.
* `experiments/make_figures/`: `figB.py` (FDFD field and adjoint check) and its style
  helpers in `common.py`. The figure scripts of the original submission were removed
  because their results were withdrawn in the revision.

## License and citation

Apache License 2.0 (see [LICENSE](LICENSE)). Please cite the papers listed in
[CITATION.cff](CITATION.cff).
