# channel.py
import numpy as np
from scipy.signal import upfirdn

def simulate_imdd_pam4_cd_pd(Nsym, SNRdB, Rsym, Ns, rolloff, span, L_km, D, lam, c=3e8, seed=None):
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
    '''
    if seed is not None:
        np.random.seed(seed)

    Fs = Rsym * Ns
    L = L_km * 1e3
    beta2 = -(lam**2 * D) / (2*np.pi*c)

    # 1) Bit generation and PAM-4 mapping (Gray)

    # Generate random bits: 2 rows (b1, b2), Nsym columns
    bits = np.random.randint(0, 2, (2, Nsym)) # b(1, :) are the MSBs AND b(2, :) are the LSBs
    pam4_levels = np.array([-3, -1, 3, 1])
    # generate the incides for mapping (00 -> idx0 -> -3, 01 -> idx1 -> -1, 11 -> idx3 -> 1, 10 -> idx2 -> 3)
    symbol_indices = 2 * bits[0, :] + bits[1, :]
    # Map to levels
    tx_symbols = pam4_levels[symbol_indices] # vector of length 1 x Nsym 

    # 2) Upsampling + RRC pulse shaping

    txRRC = rrc_filter(rolloff, span, Ns)  # Tx pulse
    txWave = upfirdn(txRRC, tx_symbols, up = Ns, down = 1) # Tx waveform

    # trim the delay introduced by the filter
    grpDelay = int(span * Ns / 2)  # group delay of the RRC filter
    txWave = txWave[grpDelay:-grpDelay]

    # 3) Optical field & chromatic dispersion (via the FFT)
    # we simply treat the txwave as the complex baseband optical field envelope
    Etx = txWave  # Optical field (real-valued for IMDD)
    N = len(Etx)
    
    f = np.fft.fftfreq(N, 1/Fs)  # Frequency grid
    H_cd = np.exp(-1j * (beta2/2) * (2 * np.pi * f)**2 * L)  # CD transfer function

    Etx_f = np.fft.fft(Etx)
    Erx_f = Etx_f * H_cd  # Apply CD in freq. domain
    Erx = np.real(np.fft.ifft(Erx_f)) # go back to time domain and take the real part.


    # 4) Square-law photodiode + AWGN
    rxElec = np.abs(Erx)**2;   # intensity detection

    # normalize signal power to 1 and add AWGN
    signal_power = np.mean(rxElec**2)
    rxElec = rxElec / np.sqrt(signal_power)  # normalize power to 1
    signal_power = np.mean(rxElec**2)  # should be 1 now
    SNR_linear = 10**(SNRdB / 10)
    noise_power = signal_power / SNR_linear
    noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(rxElec)) # real AWGN
    rxElec_noisy = rxElec + noise   

    # 5) Rx matched filter + downsampling (ideal timing)
    rxRRC = txRRC  # matched filter is same as tx RRC 
    rxMF = upfirdn(rxRRC, rxElec_noisy, up = 1, down = 1)  # matched filtering
    # trim the delay introduced by the filter
    totGrpDelay = span * Ns  # total group delay from Tx and Rx RRC filters (total = Tx + Rx)
    rxMF = rxMF[int(totGrpDelay):-int(totGrpDelay)]

    # Ideal sampling timing: sample every Ns-th sample starting from the first
    rxSyms_noEq = rxMF[::Ns] # Nsym samples (if lengths alligned correctly)

    # ensure output length is Nsym
    Nmin = min(len(rxSyms_noEq), len(tx_symbols))
    tx_symbols = tx_symbols[:Nmin]
    rxSyms_noEq = rxSyms_noEq[:Nmin]
    
    return rxSyms_noEq, tx_symbols




# later check if this is correct!!!
def rrc_filter(beta, span, Ns):
    '''
    Root Raised Cosine (RRC) filter 
    
    :param beta: roll-off factor
    :param span: filter span in symbols
    :param Ns: samples per symbol
    '''
    N_taps = span * Ns + 1
    t = np.arange(-span/2, span/2 + 1/Ns, 1/Ns)
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