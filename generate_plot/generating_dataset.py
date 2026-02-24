# generating_dataset.py
import numpy as np
from simulate.channel import simulate_imdd_pam4_cd_pd
import simulate.config as config

def make_windows(rx, tx, W):
    X, y = [], []
    for k in range(W, len(rx) - W):
        X.append(rx[k-W:k+W+1])
        y.append(tx[k])
    return np.array(X), np.array(y)

def split_dataset(X, y, seed=0):
    rng = np.random.default_rng(seed)
    N = len(X)
    idx = rng.permutation(N)

    n_train = int(config.TRAIN_FRAC * N)
    n_val   = int(config.VAL_FRAC * N)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])

def load_noise_floor():
    data = np.load(config.NOISE_FLOOR_FILE, allow_pickle=True)
    return float(data["noise_var_floor"])

def make_dataset(L_km: float, W: int, out_file: str, seed: int = 42):
    noise_var_floor = load_noise_floor()

    rx, tx, _ = simulate_imdd_pam4_cd_pd(
        Nsym=config.NSYM,
        SNRdB=config.SNR_DB,
        Rsym=config.RSYM,
        Ns=config.NS,
        rolloff=config.ROLLOFF,
        span=config.SPAN,
        L_km=L_km,
        D=config.D,
        lam=config.LAMBDA,
        c=config.C,
        seed=seed,
        noise_var_floor=noise_var_floor
    )

    X, y = make_windows(rx, tx, W)
    Xtr, ytr, Xval, yval, Xte, yte = split_dataset(X, y, seed=seed)

    np.savez(
        out_file,
        X_train=Xtr, y_train=ytr,
        X_val=Xval,  y_val=yval,
        X_test=Xte,  y_test=yte,
        meta=dict(L_km=L_km, W=W, seed=seed,)
    )
    print("Saved", out_file, "shapes:", Xtr.shape, Xval.shape, Xte.shape)