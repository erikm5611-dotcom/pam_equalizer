# noise_floor.py
import numpy as np
import config
from channel import simulate_imdd_pam4_cd_pd

def main():
    _, _, noise_var_floor = simulate_imdd_pam4_cd_pd(
        Nsym=200_000,
        SNRdB=config.SNR_DB,      # reference SNR for calibration
        Rsym=config.RSYM,
        Ns=config.NS,
        rolloff=config.ROLLOFF,
        span=config.SPAN,
        L_km=config.L_KM,         # your reference link (e.g. 0.2 km)
        D=config.D,
        lam=config.LAMBDA,
        c=config.C,
        seed=1,
        noise_var_floor=None      # IMPORTANT: compute from SNR here
    )

    np.savez(
        config.NOISE_FLOOR_FILE,
        noise_var_floor=float(noise_var_floor),
        ref_params=dict(
            SNR_DB=config.SNR_DB,
            RSYM=config.RSYM,
            NS=config.NS,
            ROLLOFF=config.ROLLOFF,
            SPAN=config.SPAN,
            L_KM=config.L_KM,
            D=config.D,
            LAMBDA=config.LAMBDA
        )
    )

    print("Saved noise_var_floor =", noise_var_floor, "to", config.NOISE_FLOOR_FILE)

if __name__ == "__main__":
    main()