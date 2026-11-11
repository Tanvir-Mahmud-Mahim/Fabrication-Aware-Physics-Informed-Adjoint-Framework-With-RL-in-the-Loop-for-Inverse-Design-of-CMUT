# Revision experiments

Scripts that produce every result of the revised article (see the repository README for the
mapping to sections, figures and tables). Run from this folder, for example:

```bash
python e0_data_audit.py
python e1_surrogate.py 40 0 1     # repeat until all 100 jobs are cached
python e2b_engine.py; python e2c_max.py; python e2d_robust.py; python e2e_noise.py 120; python e2e_noise.py 400
python e3_rl.py sac_gcn_per 0 40  # repeat until 2000 steps; likewise for the other agents/seeds
python e3b_policy.py 0 40; python e3c_eval.py
python e4_photonic.py data 0 40; python e4_photonic.py data 1 40; python e4_photonic.py grad 40
python e4_photonic.py fit; python e4_photonic.py opt nominal 40; python e4_photonic.py opt triple 40
python e4_photonic.py opt cvar 40; python e4_photonic.py yield 40
python make_revision_figures.py   # result figures (Figs. 3-7)
python make_diagrams.py           # schematic figures (Figs. 1-2)
```

`results/` holds the cached JSON outputs used by the figures and tables.
