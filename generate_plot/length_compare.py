# length_compare.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from .generating_dataset import make_dataset
from .regression_implementation_mlp import (
    standardize_fit, standardize_apply,
    MLP, train_regressor_torch, predict_regressor,
    nearest_level, compute_ber_ser
)
from .train_kan import build_kan
import torch

def run_one(dataset_file, model_type, input_dim): # one complete run of MLP or KAN on dataset, returns BER, SER and best validation MSE
    data = np.load(dataset_file, allow_pickle=True)
    Xtr, ytr = data["X_train"], data["y_train"]
    Xv,  yv  = data["X_val"],   data["y_val"]
    Xte, yte = data["X_test"],  data["y_test"]

    # standardize
    mu, sigma = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mu, sigma)
    Xv  = standardize_apply(Xv,  mu, sigma)
    Xte = standardize_apply(Xte, mu, sigma)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "MLP":
        model = MLP(input_dim=input_dim, hidden=64)
        model, best_val = train_regressor_torch(model, Xtr, ytr, Xv, yv,
                                                lr=1e-3, wd=1e-4,
                                                max_epochs=80, patience=10,
                                                device=device)
    elif model_type == "KAN":
        model = build_kan(input_dim=input_dim, hidden=16, grid=5, k=3, seed=0, device=device)
        model, best_val = train_regressor_torch(model, Xtr, ytr, Xv, yv,
                                                lr=3e-4, wd=0.0,
                                                max_epochs=80, patience=15,
                                                device=device)
    else:
        raise ValueError(model_type)

    y_pred = predict_regressor(model, Xte, device=device)
    y_hat = nearest_level(y_pred)
    ber, ser = compute_ber_ser(yte.astype(int), y_hat.astype(int))
    return ber, ser, best_val

def main():
    # sweep settings
    Ls = [ 0.1]          # 10 points from 0.1km to 2km
    Ws = [3]                         # WINDOW_HALF choices
    models = ["MLP", "KAN"]

    os.makedirs("datasets", exist_ok=True)
    out_csv = "results_L_sweep.csv"

    rows = []
    for L in Ls:
        for W in Ws:
            ds_file = f"datasets/ds_L{L:.3f}_W{W}.npz"
            if not os.path.exists(ds_file):
                make_dataset(L_km=float(L), W=int(W), out_file=ds_file, seed=42)

            input_dim = 2*W + 1

            for m in models:
                ber, ser, best_val = run_one(ds_file, m, input_dim=input_dim)
                print(f"L={L:.3f}km W={W:02d} {m}: BER={ber:.4g} SER={ser:.4g} bestValMSE={best_val:.4g}")
                rows.append([float(L), int(W), m, float(ber), float(ser), float(best_val)])

    # save CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L_km", "W_half", "model", "BER", "SER", "best_val_mse"])
        w.writerows(rows)
    print("Saved", out_csv)

    # plot BER vs L (one curve per (model,W))
    plt.figure()
    for m in models:
        for W in Ws:
            xs = [r[0] for r in rows if r[1] == W and r[2] == m]
            ys = [r[3] for r in rows if r[1] == W and r[2] == m]
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            plt.semilogy(xs, ys, marker="o", label=f"{m} W={W}")

    plt.xlabel("L[km]")
    plt.ylabel("BER")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("ber_vs_L.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()