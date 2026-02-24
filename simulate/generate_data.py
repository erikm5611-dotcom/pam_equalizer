# generate_data.py
import numpy as np
from channel import simulate_imdd_pam4_cd_pd
import config

# KAN (and any NN equalizer) does not take single symbols, it takes context in the form of a larger Window of the received symbols
def make_windows(rx, tx, W):
    """
    rx : received symbols (N,)
    tx : transmitted symbols (N,)
    W  : half window size
    """
    X = []
    y = []

    for k in range(W, len(rx) - W):
        X.append(rx[k-W:k+W+1])
        y.append(tx[k])

    return np.array(X), np.array(y)

def load_noise_floor():
    data = np.load(config.NOISE_FLOOR_FILE, allow_pickle=True)
    return float(data["noise_var_floor"])

noise_var_floor = load_noise_floor()
print("Using noise_var_floor =", noise_var_floor)

# this creates the received symbols (with CD and noise) and the transmitted symbols (without CD and noise)
def generate_raw_symbols():
    rx, tx, _ = simulate_imdd_pam4_cd_pd(
        Nsym=config.NSYM,
        SNRdB=config.SNR_DB,
        Rsym=config.RSYM,
        Ns=config.NS,
        rolloff=config.ROLLOFF,
        span=config.SPAN,
        L_km=config.L_KM,
        D=config.D,
        lam=config.LAMBDA,
        c=config.C,
        seed=config.SEED,
        noise_var_floor= noise_var_floor # add noise from noise floor that we calculate in noise_floor.py
    )
    return rx, tx

def split_dataset(X, y):
    N = len(X)
    idx = np.random.permutation(N)

    n_train = int(config.TRAIN_FRAC * N)
    n_val   = int(config.VAL_FRAC * N)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx]
    )

def main():
    np.random.seed(config.SEED)

    # 1) Channel simulation
    rx, tx = generate_raw_symbols()

    # 2) Windowing
    X, y = make_windows(rx, tx, config.WINDOW_HALF)

    # 3) Dataset split
    Xtr, ytr, Xval, yval, Xte, yte = split_dataset(X, y)

    # 4) saving
    np.savez(
        "dataset_pam4_cd_pd.npz",
        X_train=Xtr,
        y_train=ytr,
        X_val=Xval,
        y_val=yval,
        X_test=Xte,
        y_test=yte,
        params=dict(vars(config))
    )

    print("Dataset generated:")
    print(f"  Train: {Xtr.shape}")
    print(f"  Val  : {Xval.shape}")
    print(f"  Test : {Xte.shape}")

if __name__ == "__main__":
    main()
