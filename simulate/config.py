# config.py
import os 

# Modulation
M = 4
BITS_PER_SYM = 2

# Simulation size
NSYM = 200_000

# Rates
RSYM = 112e9 # (symbols per second)
NS = 4
FS = RSYM * NS

# Pulse shaping
ROLLOFF = 0.2
SPAN = 20

# Fiber / CD
L_KM = 2
D = 17e-6   # s/m^2    (same as 17 ps/(nm·km))
LAMBDA = 1550e-9
C = 3e8

# Channel / noise
SNR_DB = 25

# Equalizer window (for later)
WINDOW_HALF = 10   # -> window size = 2*WINDOW_HALF + 1

# Dataset split
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15

NOISE_FLOOR_FILE = os.path.join(os.path.dirname(__file__), "noise_floor.npz")
NOISE_VAR_FLOOR = None   # will be loaded from file if present

# Receiver electrical bandwidth (low-pass)
RX_LPF_ENABLE = True
RX_LPF_FC_HZ  = 33.6e9   # 
RX_LPF_ORDER  = 4      # Butterworth order


# Reproducibility
SEED = 42
# SEED = None for non-reproducible results