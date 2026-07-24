"""Experiment 1 (Fig. F2/F3): train the CMUT multi-physics PINN.

Usage:  python -m experiments.run_cmut_pinn [--csv path/to/fem_database.csv]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from parl_id.data.cmut_loader import load_csv, synthetic_database  # noqa: E402
from parl_id.pinn.train import make_cmut_pinn, train_cmut_pinn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="FEM database CSV (real data)")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--out", default="outputs/cmut_pinn.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_csv(args.csv) if args.csv else synthetic_database()
    print(f"database: {data['params'].shape[0]} samples | device: {device}")

    model = make_cmut_pinn(device)
    history = train_cmut_pinn(model, data, epochs=args.epochs, device=device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
