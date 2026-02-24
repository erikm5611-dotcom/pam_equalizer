import numpy as np

# Gray mapping consistent with  channel:
pam_to_bits = {-3:(0,0), -1:(0,1),  1:(1,1),  3:(1,0)}

# compute symbol error rate and bit error rate given the true transmitted symbols tx and the decoded symbols rx_hat
def ser_ber(tx, rx_hat):
    ser = np.mean(rx_hat != tx)
    tx_bits = np.array([pam_to_bits[int(s)] for s in tx], dtype=int)
    rx_bits = np.array([pam_to_bits[int(s)] for s in rx_hat], dtype=int)
    ber = np.mean(tx_bits != rx_bits)
    return ser, ber

def main():
    data = np.load("dataset_pam4_cd_pd.npz")
    Xtr, ytr = data["X_train"], data["y_train"].astype(int) # extract the received symbols and transmitted symbols from the dataset 
    Xte, yte = data["X_test"],  data["y_test"].astype(int)

    center = Xtr.shape[1] // 2 # find the center index of the window (should be 40 for a window of 81)
    rx_tr = Xtr[:, center]
    rx_te = Xte[:, center]

    # remove DC (carries no info) use training mean/std for both 
    rx_tr = rx_tr - rx_tr.mean()
    rx_te = rx_te - rx_tr.mean()
    rx_tr = rx_tr / (rx_tr.std() + 1e-12)
    rx_te = rx_te / (rx_tr.std() + 1e-12)

    levels = np.array([-3, -1, 1, 3])

    # learn mean rx for each symbol 
    # for that we compute average recevided value for each transmitted PAM level
    means = np.array([rx_tr[ytr == a].mean() for a in levels])
    order = np.argsort(means) # find the order of the levels from lowest mean rx to highest mean rx should be still mean(-3) < mean(-1) < mean(1) < mean(3)
    levels_sorted = levels[order] 
    means_sorted = means[order]

    thr = (means_sorted[:-1] + means_sorted[1:]) / 2.0 # minimum-distance decision rule (midpoint between the means of neighboring levels)

    # slice using  thresholds computed from the training data and evaluate on the test data which we did not see when we estimated the thresholds
    rx_hat = np.empty_like(yte)
    rx_hat[rx_te < thr[0]] = levels_sorted[0]
    rx_hat[(rx_te >= thr[0]) & (rx_te < thr[1])] = levels_sorted[1]
    rx_hat[(rx_te >= thr[1]) & (rx_te < thr[2])] = levels_sorted[2]
    rx_hat[rx_te >= thr[2]] = levels_sorted[3]

    SER, BER = ser_ber(yte, rx_hat)

    print("No-EQ baseline (memoryless calibrated slicer):")
    print("means per level:", dict(zip(levels, means)))
    print("level order low->high:", levels_sorted)
    print("thresholds:", thr)
    print(f"SER={SER:.4f}, BER={BER:.4f}")

if __name__ == "__main__":
    main()
