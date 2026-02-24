# channel.py
import numpy as np
from scipy.signal import upfirdn, butter, lfilter
from . import config

def simulate_imdd_pam4_cd_pd(Nsym, SNRdB, Rsym, Ns, rolloff, span, L_km, D, lam, c=3e8, seed=None, noise_var_floor=None, RX_LPF_ENABLE= config.RX_LPF_ENABLE, RX_LPF_FC_HZ=config.RX_LPF_FC_HZ, RX_LPF_ORDER=config.RX_LPF_ORDER):
    '''
    This function simulates an IMDD PAM-4 system with chromatic dispersion and
    square-law photodiode detection. Without any equalization at the receiver (happens later).
    
    :param Nsym: Number of symbols to simulate
    :param SNRdB: Signal-to-noise ratio in dB
    :param Rsym: Symbol rate (baud)
    :param Ns: Samples per symbol
    :param rolloff: rolloff factor of the RRC filter
    :param span: filter span in symbols
    :param L_km: length of the fiber in km
    :param D: dispersion parameter in ps/(nm*km)
    :param lam: wavelength in meters
    :param c: speed of light in m/s (default: 3e8)
    :param seed: random seed to make it reproducable (default: None)
    :param noise_var_floor: if not None, use this fixed noise variance instead of calculating it from the SNR
    '''
    if seed is not None:
        np.random.seed(seed)

    Fs = Rsym * Ns
    L = L_km * 1e3
    beta2 = -(lam**2 * D) / (2*np.pi*c)


    # 1) Bit generation and PAM-4 mapping (Gray)
    bits = np.random.randint(0, 2, (2, Nsym))
    b0 = bits[0, :]  # MSB
    b1 = bits[1, :]  # LSB

    # Gray mapping: 00->-3, 01->-1, 11->+1, 10->+3
    tx_symbols = np.empty(Nsym)
    tx_symbols[(b0==0) & (b1==0)] = -3
    tx_symbols[(b0==0) & (b1==1)] = -1
    tx_symbols[(b0==1) & (b1==1)] =  1
    tx_symbols[(b0==1) & (b1==0)] =  3

    # 2) Upsampling + RRC pulse shaping

    txRRC = rrc_filter(rolloff, span, Ns)  # Tx pulse
    txWave = upfirdn(txRRC, tx_symbols, up = Ns, down = 1) # Tx waveform

    # trim the delay introduced by the filter
    grpDelay = span * Ns // 2  # group delay of the RRC filter
    txWave = txWave[grpDelay : grpDelay + Nsym * Ns]

   # 3) Optical field & chromatic dispersion (via the FFT)
    alpha = 1.0
    peak = np.max(np.abs(txWave))
    bias = 1.2 * alpha * peak   # 20% margin above the peak to ensure I is not negative (important for IM/DD)

    Itx = bias + alpha * txWave
    Itx = np.maximum(Itx, 1e-6) # I > 0 in IM/DD + a small margin to avoid numerical issues

    Etx = np.sqrt(Itx)                  # field amplitude

    N = len(Etx)
    
    f = np.fft.fftfreq(N, 1/Fs)  # Frequency grid
    H_cd = np.exp(-1j * (beta2/2) * (2 * np.pi * f)**2 * L)  # CD transfer function

    Etx_f = np.fft.fft(Etx)
    Erx_f = Etx_f * H_cd  # Apply CD in freq. domain
    Erx = np.fft.ifft(Erx_f) # go back to time domain 

    # 4) Square-law photodiode + AWGN
    rxElec = np.abs(Erx)**2;   # intensity detection

    if noise_var_floor is None:
        # old behavior: choose noise for desired SNR at *this* operating point
        rx_ac = rxElec - np.mean(rxElec)
        sig_var = np.mean(rx_ac**2)
        SNR_linear = 10**(SNRdB/10)
        noise_var = sig_var / SNR_linear
    else:
        # fixed noise floor
        noise_var = float(noise_var_floor)

    noise = np.sqrt(noise_var) * np.random.randn(len(rxElec))
    rxElec_noisy = rxElec + noise

    # Lowpass filter to make the channel banlimited
    if RX_LPF_ENABLE:
        fc = config.RX_LPF_FC_HZ
        wn = fc / (Fs/2)          # normalized cutoff (0 - 1)

        b, a = butter(RX_LPF_ORDER, wn, btype="low")
        rxElec_noisy = lfilter(b, a, rxElec_noisy)

    # 5) Rx matched filter + downsampling (ideal timing)
    rxRRC = txRRC  # matched filter is same as tx RRC 
    rxMF = upfirdn(rxRRC, rxElec_noisy, up = 1, down = 1)  # matched filtering
    # trim the delay introduced by the filter
    totGrpDelay = span * Ns  # total group delay from Tx and Rx RRC filters (total = Tx + Rx)
    rxMF = rxMF[totGrpDelay : totGrpDelay + Nsym * Ns]

    # Ideal sampling timing: sample every Ns-th sample starting from the first
    rxSyms_noEq = rxMF[::Ns] # Nsym samples (if lengths alligned correctly)
    tx_symbols  = tx_symbols[:len(rxSyms_noEq)]

    rxSyms_noEq, tx_symbols, lag = align_symbols(rxSyms_noEq, tx_symbols, max_lag=2*span) # align rx and tx symbols by finding the lag that maximizes correlation 
    print("Best symbol lag =", lag)
    print("corr(rx,tx) =", np.corrcoef(rxSyms_noEq, tx_symbols)[0,1])
    

    return rxSyms_noEq, tx_symbols, noise_var


def rrc_filter(beta, span, Ns):
    '''
    Root Raised Cosine (RRC) filter 
    
    :param beta: roll-off factor
    :param span: filter span in symbols
    :param Ns: samples per symbol
    '''
    N = span * Ns
    t = np.arange(-N//2, N//2 + 1) / Ns
    rrc = np.zeros_like(t)

    for i in range(len(t)):
        if np.isclose(t[i], 0.0):
            rrc[i] = 1.0 - beta + (4 * beta / np.pi)
        elif beta != 0 and np.isclose(abs(t[i]), 1/(4*beta)):
            rrc[i] = (beta / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*beta)) + (1 - 2/np.pi) * np.cos(np.pi/(4*beta)))
        else:
            numerator = np.sin(np.pi * t[i] * (1 - beta)) + 4 * beta * t[i] * np.cos(np.pi * t[i] * (1 + beta))
            denominator = np.pi * t[i] * (1 - (4 * beta * t[i])**2)
            rrc[i] = numerator / denominator

    rrc /= np.sqrt(np.sum(rrc**2))  # Normalize energy
    return rrc

def align_symbols(rx, tx, max_lag=200):
    """
    Align rx to tx by finding lag that maximizes correlation.
    Searches lag in [-max_lag, +max_lag] symbols.
    Returns (rx_aligned, tx_aligned, best_lag).
    """
    rx0 = rx - np.mean(rx)
    tx0 = tx - np.mean(tx)

    best_lag = 0
    best_val = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = rx0[lag:]
            t = tx0[:len(r)]
        else:
            r = rx0[:lag]
            t = tx0[-lag:][:len(r)]

        if len(r) < 5000:
            continue

        val = np.dot(r, t) / len(r)
        if val > best_val:
            best_val = val
            best_lag = lag

    # apply lag
    if best_lag >= 0:
        rx_a = rx[best_lag:]
        tx_a = tx[:len(rx_a)]
    else:
        rx_a = rx[:best_lag]
        tx_a = tx[-best_lag:][:len(rx_a)]

    return rx_a, tx_a, best_lag