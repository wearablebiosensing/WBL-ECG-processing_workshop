# %%
# %% [markdown]
# # PhysioNet CINC 2017 Inference Demo
# This notebook downloads a sample from the PhysioNet CINC 2017 dataset, adds optional augmentations (noise, baseline wander, etc.),
# runs it through the best performing teacher model, and visualizes the reconstruction and R-peak detection results.
# It also compares the model's R-peak detection against the standard Pan-Tompkins algorithm on the noisy signal.

# %% [markdown]
# ## 1. Setup & Imports
# Ensure you have installing `wfdb`, `neurokit2`, `wget` beforehand
# `pip install wfdb neurokit2 wget matplotlib torch scipy numpy`

# %%
import os
import urllib.request
import zipfile
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import torch
import torch.nn.functional as F
from scipy import signal as sp_signal
import neurokit2 as nk

import sys
from pathlib import Path

# Add the src directory to Python path so we can import our modules
src_path = str(Path().resolve().parent / 'src_RPnet')

sys.path.append(src_path) # Assuming running from notebooks/ directory
# If running from root, just import directly
try:
    from src.model import TeacherModel
    from src.metrics import dt_to_peaks
except ImportError:
    from model import TeacherModel
    from metrics import dt_to_peaks

print(plt.style.available)


# %%

# %%
# Model_name = "latest.pt"
Model_name = "best.pt"
ckpt_path = "/work/pi_kunalm_uri_edu/Arm_ECG_RPnet/checkpoints/teacherV3-FPfix-53796151/teacher-V3-FPfix-53796151_20260322_163511/"+ Model_name # Update this path if different
DATA_DIR = "/work/pi_kunalm_uri_edu/Arm_ECG_RPnet/raw_data/physionet_2017"
os.makedirs(DATA_DIR, exist_ok=True)

# URL for a single training set archive (training2017.zip contains A00001.mat etc.)
ZIP_URL = "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
ZIP_FILE = os.path.join(DATA_DIR, "training2017.zip")

if not os.path.exists(ZIP_FILE):
    print("Downloading PhysioNet CINC 2017 Training Data (~160MB)...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_FILE)
    print("Download complete.")
    
    print("Extracting...")
    with zipfile.ZipFile/(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")
else:
    print("Dataset already downloaded and extracted.")

RECORD_DIR = os.path.join(DATA_DIR, "training2017")
# Pick a random record
mat_files = glob.glob(os.path.join(RECORD_DIR, "*.mat"))
if not mat_files:
    raise FileNotFoundError(f"No .mat files found in {RECORD_DIR}")

record_name = os.path.basename(random.choice(mat_files)).replace(".mat", "")
record_path = os.path.join(RECORD_DIR, record_name)

print(f"=====================================")
print(f"Randomly selected record: {record_name}")
print(f"=====================================")

# %%

# =============================================================================
# Randomized Arm-ECG Augmentation
# PhysioNet CINC 2017 is standard lead II (~1-2 mV P-P). Arm/wrist ECG is
# much weaker (~0.1-0.8 mV P-P) and has stronger baseline wander from movement.
# =============================================================================
rng = np.random.default_rng()  # fresh random state each run
# ## 3. Data Loading, Augmentation & R-Peak Reference
# We load the record, add noise, and compute reference R-peaks.

# %%
record = wfdb.rdrecord(record_path)
fs = record.fs
sig = record.p_signal[:, 0]  # Take first channel

# Take a 5-second window (500 samples at 100Hz later)
# Assuming 5 seconds fits easily (CINC signals are usually 30-60s)
start_sec = 2.0
length_sec = 5.0
start_idx = int(start_sec * fs)
end_idx = int((start_sec + length_sec) * fs)
clean_window = sig[start_idx:end_idx]

# Resample to 100 Hz if necessary (CINC 2017 is typically 300Hz)
if fs != 100:
    resample_len = int(len(clean_window) * 100 / fs)
    clean_window = sp_signal.resample(clean_window, resample_len)
    fs = 100.0


# --- 1. Amplitude rescaling to realistic arm-ECG range ---
target_amplitude_mv = rng.uniform(0.0001, 0.005)   # arm ECG: 0.002-0.8 mV peak-to-peak
current_pp = np.ptp(clean_window)
scale_factor = target_amplitude_mv / max(current_pp, 1e-6)
clean_window = clean_window * scale_factor

# --- 2. Multi-component baseline wander ---
# Realistic sources: respiration (0.15-0.4 Hz), slow body movement (0.05-0.15 Hz),
# postural drift (0.01-0.05 Hz), and occasional faster motion artifact (0.4-0.8 Hz)
n_bw = rng.integers(3, 7)   # 3-6 independent sinusoidal components
t = np.arange(500) / fs
bw_signal = np.zeros(500)
bw_components = []
for _ in range(n_bw):
    freq  = rng.uniform(0.01, 0.80)                              # Hz
    amp   = rng.uniform(0.05, 0.40) * target_amplitude_mv       # scale with signal
    phase = rng.uniform(0.0, 2 * np.pi)
    bw_signal += amp * np.sin(2 * np.pi * freq * t + phase)
    bw_components.append((freq, amp, phase))

# --- 3. Gaussian noise at random SNR (10-24 dB) ---
snr_db = rng.uniform(0.00001, 0.05)
snr_db = 0.000001
sig_power = np.mean(clean_window ** 2)
noise_std = np.sqrt(sig_power / (10 ** (snr_db / 10)))
gaussian_noise = rng.normal(0, noise_std, 500)

# Compose final noisy signal
noisy_window = clean_window + bw_signal + 1.5*gaussian_noise

# --- Print augmentation summary ---
print("=" * 52)
print("  AUGMENTATION PARAMETERS")
print("=" * 52)
print(f"  Amplitude rescaling")
print(f"    Original P-P  : {current_pp:.4f} mV")
print(f"    Target P-P    : {target_amplitude_mv:.4f} mV")
print(f"    Scale factor  : {scale_factor:.4f}x")
print(f"  Baseline wander ({n_bw} components)")
for i, (f, a, ph) in enumerate(bw_components):
    print(f"    [{i+1}] freq={f:.3f} Hz  amp={a:.4f} mV  phase={np.degrees(ph):.1f}°")
bw_pp = np.ptp(bw_signal)
print(f"    Total BW P-P  : {bw_pp:.4f} mV  ({100*bw_pp/target_amplitude_mv:.1f}% of signal)")
print(f"  Gaussian noise")
print(f"    SNR           : {snr_db:.5f} dB")
print(f"    Noise std     : {noise_std:.5f} mV")
print("=" * 52)

# Extract Reference R-peaks (Ground Truth) from the CLEAN window
try:
    _, rpeaks = nk.ecg_peaks(clean_window, sampling_rate=100)
    true_peaks = rpeaks["ECG_R_Peaks"]
except Exception:
    # Fallback to simple find_peaks if neurokit2 fails
    true_peaks, _ = sp_signal.find_peaks(clean_window, distance=int(100*0.4), height=np.mean(clean_window))

# Generate Ground Truth Distance Transform map (clipped to 50 to match training)
y_dt = np.full(len(clean_window), 50.0)
for p in true_peaks:
    dists = np.abs(np.arange(len(clean_window)) - p)
    y_dt = np.minimum(y_dt, dists)
y_dt = np.clip(y_dt, 0.0, 50.0)

# %% [markdown]
# ## 4. Benchmark: Classical R-peak Detectors on Noisy Data

# %%
def _run_nk2(sig, method):
    try:
        _, info = nk.ecg_peaks(sig, sampling_rate=100, method=method)
        return np.array(info["ECG_R_Peaks"])
    except Exception as e:
        print(f"  [{method}] failed: {e}")
        return np.array([], dtype=int)

def _run_xqrs(sig, fs):
    try:
        import wfdb.processing
        xqrs = wfdb.processing.XQRS(sig=sig.astype(np.float64), fs=int(fs))
        xqrs.detect(verbose=False)
        return np.array(xqrs.qrs_inds)
    except Exception as e:
        print(f"  [XQRS] failed: {e}")
        return np.array([], dtype=int)

def match_peaks(detected, reference, tol=5):
    """TP/FP/FN with ±tol sample tolerance window (50 ms at 100 Hz)."""
    detected  = np.array(sorted(detected),  dtype=int)
    reference = np.array(sorted(reference), dtype=int)
    matched = set()
    tp = 0
    for d in detected:
        if len(reference) == 0:
            break
        diffs = np.abs(reference - d)
        best = int(np.argmin(diffs))
        if diffs[best] <= tol and best not in matched:
            tp += 1
            matched.add(best)
    fp = len(detected) - tp
    fn = len(reference) - tp
    se  = tp / max(tp + fn, 1)
    ppv = tp / max(tp + fp, 1)
    f1  = 2 * se * ppv / max(se + ppv, 1e-9)
    return {"TP": tp, "FP": fp, "FN": fn, "Se": se, "PPV": ppv, "F1": f1}

pt_peaks     = _run_nk2(noisy_window, "pantompkins1985")
nk2_peaks    = _run_nk2(noisy_window, "neurokit")
promac_peaks = _run_nk2(noisy_window, "promac")
xqrs_peaks   = _run_xqrs(noisy_window, fs)

# %% [markdown]
# ## 5. Model Inference
# Load TeacherModel and run prediction. Will use GPU if available, else CPU.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running model on: {device}")

# Load model
model = TeacherModel(base_ch=32, enable_recon=True, enable_sqi=False).to(device)
# ckpt_path = "../checkpoints/teacher_best.pth" # Update this path if different
if not os.path.exists(ckpt_path):
    print(f"Warning: Checkpoint not found at {ckpt_path}. Untrained weights will be used for demonstration.")
else:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Prepare input (Normalize)
mu = np.mean(noisy_window)
sigma = np.std(noisy_window) + 1e-8
x_in = (noisy_window - mu) / sigma

# Inference
with torch.no_grad():
    x_tensor = torch.from_numpy(x_in).unsqueeze(0).unsqueeze(0).float().to(device) # (1, 1, 500)
    preds = model(x_tensor)
    
    pred_dt = preds["dt_map"].squeeze().cpu().numpy()
    pred_recon = preds["recon"].squeeze().cpu().numpy()

# Denormalize reconstruction
pred_recon_denorm = (pred_recon * sigma) + mu

# Extract Predicted R-peaks from Model Distance Transform
# threshold < 5.0 means within 50ms. prominence >= 5.0 rejects shallow AI ripples/edge bounces.
predicted_peaks = dt_to_peaks(pred_dt, threshold=5.0, min_distance=20, prominence=5.0, fs=100.0)

# %% [markdown]
# ## 6. Visualization

# %%
# Detector registry — colour / marker / label for each method
DETECTORS = [
    {"name": "Teacher Model", "peaks": predicted_peaks, "color": "crimson",    "marker": "o", "ls": "-",  "lw": 2.0},
    {"name": "Pan-Tompkins",  "peaks": pt_peaks,        "color": "royalblue",  "marker": "s", "ls": "--", "lw": 1.5},
    {"name": "NK2 Default",   "peaks": nk2_peaks,       "color": "darkorange", "marker": "D", "ls": "-.", "lw": 1.5},
    {"name": "NK2 PromAC",    "peaks": promac_peaks,    "color": "mediumpurple","marker": "P", "ls": ":",  "lw": 1.5},
    {"name": "WFDB XQRS",    "peaks": xqrs_peaks,      "color": "teal",       "marker": "*", "ls": "--", "lw": 1.5},
]

plt.style.use('seaborn-darkgrid')
# plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
# plt.style.use("fivethirtyeight")
# plt.style.use("cyberpunk")

time_axis = np.arange(len(clean_window)) / 100.0

# ── Figure 1: Reconstruction + All detectors ─────────────────────────────────
fig1, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 3]})

ax1.plot(time_axis, noisy_window,      color="gray",       alpha=0.5, lw=1,   ls="--", label="Noisy Input")
ax1.plot(time_axis, clean_window,      color="black",      lw=1.5,            label="Clean Reference")
ax1.plot(time_axis, pred_recon_denorm, color="dodgerblue", lw=2,              label="Teacher Reconstruction")
ax1.set_title(
    f"Record {record_name}  |  Arm-ECG {target_amplitude_mv:.2f} mV P-P  |  "
    f"SNR {snr_db:.1f} dB  |  BW {n_bw} components",
    fontsize=11, fontweight="bold"
)
ax1.set_ylabel("Amplitude (mV)")
ax1.legend(loc="upper right", fontsize=9)

ax3.plot(time_axis, noisy_window, color="black", lw=1, alpha=0.6, label="Noisy Signal")
for p in true_peaks:
    ax3.axvline(p / 100.0, color="green", lw=2, alpha=0.35)
ax3.plot([], [], color="green", lw=2, alpha=0.5, label="GT R-peak")

sig_range = np.ptp(noisy_window)
y_offsets = np.linspace(sig_range * 0.10, sig_range * 0.55, len(DETECTORS))
for det, y_off in zip(DETECTORS, y_offsets):
    peaks = det["peaks"]
    if len(peaks) == 0:
        ax3.plot([], [], color=det["color"], marker=det["marker"],
                 ls="None", label=f"{det['name']} (0 peaks)")
        continue
    valid = peaks[(peaks >= 0) & (peaks < len(noisy_window))]
    y_pos = np.max(noisy_window) + y_off
    ax3.scatter(valid / 100.0, np.full(len(valid), y_pos),
                color=det["color"], marker=det["marker"], s=80, zorder=5,
                label=f"{det['name']} ({len(valid)})")
    for vp in valid:
        ax3.axvline(vp / 100.0, color=det["color"], lw=0.8, alpha=0.25, ls=det["ls"])

ax3.set_title("R-Peak Detection: All Detectors", fontsize=11, fontweight="bold")
ax3.set_xlabel("Time (seconds)")
ax3.set_ylabel("Amplitude (mV)")
ax3.legend(loc="upper right", fontsize=8, ncol=2)

fig1.tight_layout()
plt.show()

# ── Figure 2: Distance Transform ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(1, 1, figsize=(15, 4))

ax2.plot(time_axis, y_dt,    color="black",   lw=1.5, ls="--", alpha=0.6, label="GT DT")
ax2.plot(time_axis, pred_dt, color="crimson", lw=2,                       label="Predicted DT")
for p in true_peaks:
    ax2.axvline(p / 100.0, color="green", lw=1.5, alpha=0.5, ls="-")
ax2.plot([], [], color="green", lw=1.5, label="GT R-peak")
ax2.set_title("Distance Transform: Ground Truth vs Teacher Model", fontsize=11, fontweight="bold")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("DT (samples)")
ax2.set_ylim(-2, 55)
ax2.legend(loc="upper right", fontsize=9)

fig2.tight_layout()
plt.show()

# ── Metrics table ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 62}")
print(f"  DETECTOR COMPARISON  (tolerance ±5 samples = ±50 ms)")
print(f"  GT peaks: {len(true_peaks)}  at {list(true_peaks)}")
print(f"{'=' * 62}")
print(f"  {'Detector':<18} {'N':>4}  {'TP':>4} {'FP':>4} {'FN':>4}  {'Se':>6} {'PPV':>6} {'F1':>6}")
print(f"  {'-'*58}")
for det in DETECTORS:
    m = match_peaks(det["peaks"], true_peaks, tol=5)
    print(
        f"  {det['name']:<18} {len(det['peaks']):>4}  "
        f"{m['TP']:>4} {m['FP']:>4} {m['FN']:>4}  "
        f"{m['Se']:>6.3f} {m['PPV']:>6.3f} {m['F1']:>6.3f}"
    )
print(f"{'=' * 62}")
print(f"\n  AUGMENTATION")
print(f"  Arm-ECG amp   : {target_amplitude_mv:.3f} mV P-P  (scale {scale_factor:.3f}x)")
print(f"  Baseline wand : {n_bw} components, P-P {np.ptp(bw_signal):.3f} mV")
print(f"  SNR           : {snr_db:.1f} dB")
# %%


# %%

# %% [markdown]
# ## 7. Interpretability: Input Saliency Analysis
#
# Three complementary attribution methods, each with a different bias:
#   - **Gradient saliency**     — fast, local; shows where the loss surface is steep
#   - **Integrated Gradients**  — path-integral from silent baseline; satisfies completeness axiom
#   - **Occlusion sensitivity** — perturbation-based; most interpretable to non-ML readers
#
# Unique to the DT regression formulation: we run IG *per detected beat*, so we can ask
# "what did the model look at to find THIS specific beat?" rather than only global attribution.

# %%
import matplotlib.colors as mcolors
import matplotlib.cm as cm

model.eval()  # ensure BN uses running stats, not batch stats

# Helper: valid predicted peaks clipped to signal bounds
_vpks = predicted_peaks[(predicted_peaks >= 0) & (predicted_peaks < 500)] if len(predicted_peaks) > 0 else np.array([], dtype=int)

def _scalar_target(out, peak_indices):
    """Mean DT at peak locations (lower = more confident peak). Falls back to global mean."""
    dt = out["dt_map"].squeeze()
    if len(peak_indices) > 0:
        return dt[torch.tensor(peak_indices, device=dt.device)].mean()
    return dt.mean()

# ── Method 1: Vanilla gradient saliency ──────────────────────────────────────
def gradient_saliency(model, x_tensor, peak_indices):
    x = x_tensor.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        _scalar_target(model(x), peak_indices).backward()
    return np.abs(x.grad.squeeze().cpu().numpy())

# ── Method 2: Integrated Gradients (zero baseline = silent signal) ───────────
def integrated_gradients(model, x_tensor, peak_indices, n_steps=64):
    baseline = torch.zeros_like(x_tensor)
    ig = torch.zeros(x_tensor.shape[-1])
    alphas = np.linspace(0.0, 1.0, n_steps + 1)
    for alpha in alphas:
        interp = (baseline + alpha * (x_tensor - baseline)).clone().detach().requires_grad_(True)
        with torch.enable_grad():
            _scalar_target(model(interp), peak_indices).backward()
        ig += interp.grad.squeeze().cpu()
    ig = ig / (n_steps + 1)
    ig = ig * (x_tensor - baseline).squeeze().cpu()
    return ig.numpy()   # signed: positive = helps detect peaks

# ── Method 3: Occlusion sensitivity ──────────────────────────────────────────
def occlusion_sensitivity(model, x_tensor, peak_indices, window=25, step=5):
    sig_len = x_tensor.shape[-1]
    sensitivity = np.zeros(sig_len)
    counts      = np.zeros(sig_len)
    with torch.no_grad():
        base_score = _scalar_target(model(x_tensor), peak_indices).item()
    for start in range(0, sig_len - window + 1, step):
        end = start + window
        x_occ = x_tensor.clone()
        x_occ[0, 0, start:end] = 0.0
        with torch.no_grad():
            occ_score = _scalar_target(model(x_occ), peak_indices).item()
        delta = abs(occ_score - base_score)
        sensitivity[start:end] += delta
        counts[start:end] += 1
    return sensitivity / np.maximum(counts, 1)

# ── Method 4: Per-beat Integrated Gradients ───────────────────────────────────
def per_beat_ig(model, x_tensor, peak_indices, n_steps=64):
    """Run IG separately for each detected peak — unique to the DT formulation."""
    results = {}
    for p in peak_indices:
        ig = integrated_gradients(model, x_tensor, [p], n_steps=n_steps)
        results[int(p)] = ig
    return results

print("Running saliency methods...", flush=True)
sal_grad = gradient_saliency(model, x_tensor, _vpks)
sal_ig   = integrated_gradients(model, x_tensor, _vpks)
sal_occ  = occlusion_sensitivity(model, x_tensor, _vpks)
sal_per_beat = per_beat_ig(model, x_tensor, _vpks) if len(_vpks) > 0 else {}
print(f"  Done. ({len(_vpks)} peaks analysed)", flush=True)

# ── Figure 3: Saliency overview (4 panels) ───────────────────────────────────
fig3, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 2, 2]})
time_axis = np.arange(500) / 100.0

# Panel A: signal coloured by gradient magnitude
ax_sig = axes[0]
sal_norm = (sal_grad - sal_grad.min()) / (sal_grad.max() - sal_grad.min() + 1e-9)
for i in range(len(time_axis) - 1):
    ax_sig.fill_between(time_axis[i:i+2], noisy_window[i:i+2],
                         alpha=0.0)   # invisible fill placeholder
# scatter with colormap gives per-sample colour
sc = ax_sig.scatter(time_axis, noisy_window, c=sal_norm, cmap="hot", s=6, zorder=3)
ax_sig.plot(time_axis, noisy_window, color="black", lw=0.6, alpha=0.3, zorder=2)
for p in true_peaks:
    ax_sig.axvline(p / 100.0, color="green", lw=1.5, alpha=0.5)
for p in _vpks:
    ax_sig.axvline(p / 100.0, color="red", lw=1.5, alpha=0.5, ls="--")
plt.colorbar(sc, ax=ax_sig, label="Gradient magnitude (norm.)", pad=0.01)
ax_sig.set_ylabel("Amplitude (mV)")
ax_sig.set_title(
    "Gradient Saliency — signal coloured by |∂DT_peaks / ∂input|  "
    "(hot = high influence on peak prediction)",
    fontsize=10, fontweight="bold"
)

# Panel B: Integrated Gradients (signed bar chart)
ax_ig = axes[1]
colors_ig = ["#c0392b" if v > 0 else "#2980b9" for v in sal_ig]
ax_ig.bar(time_axis, sal_ig, width=0.01, color=colors_ig, alpha=0.8)
ax_ig.axhline(0, color="black", lw=0.8, ls="--")
for p in true_peaks:
    ax_ig.axvline(p / 100.0, color="green", lw=1.5, alpha=0.4)
for p in _vpks:
    ax_ig.axvline(p / 100.0, color="red", lw=1.5, alpha=0.4, ls="--")
ax_ig.set_ylabel("IG attribution")
ax_ig.set_title(
    "Integrated Gradients (zero baseline → input)  "
    "Red = pushes DT down (supports peak)  |  Blue = pushes DT up (suppresses peak)",
    fontsize=10, fontweight="bold"
)

# Panel C: Occlusion sensitivity
ax_occ = axes[2]
ax_occ.fill_between(time_axis, sal_occ, alpha=0.6, color="darkorange", label="Occlusion Δ")
ax_occ.plot(time_axis, sal_occ, color="darkorange", lw=1)
for p in true_peaks:
    ax_occ.axvline(p / 100.0, color="green", lw=1.5, alpha=0.4)
for p in _vpks:
    ax_occ.axvline(p / 100.0, color="red", lw=1.5, alpha=0.4, ls="--")
ax_occ.set_ylabel("ΔΔDT (occluded)")
ax_occ.set_title(
    f"Occlusion Sensitivity (window=25 smp / 250 ms, step=5 smp)  "
    "Peaks = regions the model relies on most",
    fontsize=10, fontweight="bold"
)

# Panel D: Per-beat IG overlay
ax_pb = axes[3]
ax_pb.plot(time_axis, noisy_window, color="black", lw=0.8, alpha=0.4, label="Signal")
beat_colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(sal_per_beat), 1)))
for (peak_idx, ig_vals), col in zip(sal_per_beat.items(), beat_colors):
    ig_abs = np.abs(ig_vals)
    ig_norm = ig_abs / (ig_abs.max() + 1e-9)
    ax_pb.fill_between(time_axis, ig_norm * np.ptp(noisy_window) * 0.5 + np.min(noisy_window),
                        np.min(noisy_window), alpha=0.35, color=col,
                        label=f"Beat @{peak_idx/100:.2f}s")
    ax_pb.axvline(peak_idx / 100.0, color=col, lw=2, alpha=0.7, ls=":")
for p in true_peaks:
    ax_pb.axvline(p / 100.0, color="green", lw=1.5, alpha=0.35)
ax_pb.set_ylabel("IG (per beat)")
ax_pb.set_xlabel("Time (seconds)")
ax_pb.set_title(
    "Per-Beat Integrated Gradients — each colour = attribution field for one detected beat\n"
    "(unique to DT formulation: answers 'what did the model look at to find THIS beat?')",
    fontsize=10, fontweight="bold"
)
ax_pb.legend(loc="upper right", fontsize=7, ncol=4)

fig3.suptitle(
    f"Model Interpretability  |  Record {record_name}  |  "
    f"{len(_vpks)} predicted peaks  |  Green=GT  Red-dashed=Model",
    fontsize=11, fontweight="bold", y=1.01
)
fig3.tight_layout()
plt.show()
# %%


# %%


# %%



