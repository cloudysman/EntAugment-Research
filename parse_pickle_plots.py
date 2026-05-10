"""
parse_pickle_plots.py — Generate all paper figures from pickle dumps.

Plots:
  1. Magnitude evolution (mean magnitude per epoch, 4 methods)
  2. Per-class magnitude distribution (CIFAR-100, end of training)
  3. cos(g_clean, g_aug) evolution — augmentation fidelity (KEY METRIC)
  4. Training dynamics (loss + accuracy)

Usage:
    python parse_pickle_plots.py --dump_dir logs_dump --output_dir plots
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dump_dir', type=str, default='logs_dump')
parser.add_argument('--output_dir', type=str, default='plots')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--smooth_window', type=int, default=10,
                    help='Moving average window for smoothing')
args = parser.parse_args()

DUMP_DIR = Path(args.dump_dir)
OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    'EntAugment_CutMix',
    'RandAugment_CutMix',
    'EntAugment_only',
    'RandAugment_only',
]

# Color/style scheme — colorblind-friendly, paper-ready
STYLES = {
    'EntAugment_CutMix':    {'color': '#D62728', 'ls': '-',  'lw': 2.2, 'label': 'EntAugment + CutMix (Ours)'},
    'RandAugment_CutMix':   {'color': '#1F77B4', 'ls': '--', 'lw': 1.8, 'label': 'RandAugment + CutMix'},
    'EntAugment_only':      {'color': '#FF7F0E', 'ls': '-.', 'lw': 1.8, 'label': 'EntAugment only'},
    'RandAugment_only':     {'color': '#2CA02C', 'ls': ':',  'lw': 1.8, 'label': 'RandAugment only'},
}

# Set publication-quality font/style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


def smooth(arr, window):
    if window <= 1 or len(arr) < window:
        return np.array(arr)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def load_method_data(method, seed):
    """Load all epoch pickles for a method."""
    folder = DUMP_DIR / f"{method}_seed{seed}"
    if not folder.exists():
        print(f"⚠️  Missing: {folder}")
        return None

    pkl_files = sorted(folder.glob('epoch_*.pkl'))
    print(f"  {method}: {len(pkl_files)} epoch dumps")

    data = {
        'epochs': [],
        'cos_clean_aug_per_epoch': [],     # mean cos per epoch
        'cos_clean_aug_all': [],            # all batch values flat (for variance)
        'magnitude_mean': [],
        'magnitude_per_sample_final': None,
        'labels': None,
        'train_loss': [],
        'train_acc': [],
    }

    for pkl in pkl_files:
        with open(pkl, 'rb') as f:
            d = pickle.load(f)

        data['epochs'].append(d['epoch'])
        data['train_loss'].append(d['train_loss'])
        data['train_acc'].append(d['train_acc'])

        # cos_clean_aug
        cs = d['cos_g_clean_aug']
        if len(cs) > 0:
            data['cos_clean_aug_per_epoch'].append(np.mean(cs))
        else:
            data['cos_clean_aug_per_epoch'].append(np.nan)

        # magnitude
        if d['magnitude_per_sample'] is not None:
            data['magnitude_mean'].append(np.mean(d['magnitude_per_sample']))
        else:
            data['magnitude_mean'].append(np.nan)

    # Last epoch data
    last_pkl = pkl_files[-1]
    with open(last_pkl, 'rb') as f:
        d_last = pickle.load(f)
    data['magnitude_per_sample_final'] = d_last['magnitude_per_sample']
    data['labels'] = d_last['labels']

    return data


# ─────────────────────────────────────────────
# Load all methods
# ─────────────────────────────────────────────

print("=" * 60)
print("Loading pickle dumps...")
print("=" * 60)
all_data = {}
for m in METHODS:
    d = load_method_data(m, args.seed)
    if d is not None:
        all_data[m] = d


# ─────────────────────────────────────────────
# PLOT 1: Magnitude evolution
# ─────────────────────────────────────────────

print("\n[Plot 1] Magnitude evolution...")
fig, ax = plt.subplots(figsize=(7, 4.5))

for method in METHODS:
    if method not in all_data:
        continue
    d = all_data[method]
    if d['magnitude_per_sample_final'] is None:
        continue  # RandAugment methods don't have magnitude

    epochs = np.array(d['epochs'])
    mags = np.array(d['magnitude_mean'])

    # Smooth
    if len(mags) >= args.smooth_window:
        mags_s = smooth(mags, args.smooth_window)
        epochs_s = epochs[len(epochs) - len(mags_s):]
    else:
        mags_s = mags
        epochs_s = epochs

    style = STYLES[method]
    ax.plot(epochs_s, mags_s,
            color=style['color'], linestyle=style['ls'],
            linewidth=style['lw'], label=style['label'])

# RandAugment "magnitude" reference line (m=9, normalized to [0,1] = 9/30 = 0.3)
ax.axhline(y=9/30, color='gray', linestyle=':', linewidth=1.2,
           alpha=0.5, label='RandAugment (fixed m=9/30)')

ax.set_xlabel('Epoch')
ax.set_ylabel('Mean Augmentation Magnitude')
ax.set_title('Magnitude Evolution: Adaptive vs. Fixed')
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax.set_ylim([0, 1.0])

plt.tight_layout()
out_path = OUT_DIR / 'plot1_magnitude_evolution.png'
plt.savefig(out_path)
plt.close()
print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
# PLOT 2: Per-class magnitude distribution
# ─────────────────────────────────────────────

print("\n[Plot 2] Per-class magnitude distribution (final epoch)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

ent_methods = [m for m in METHODS if 'EntAugment' in m]

for ax, method in zip(axes, ent_methods):
    if method not in all_data:
        continue
    d = all_data[method]
    if d['magnitude_per_sample_final'] is None:
        continue

    mags = d['magnitude_per_sample_final']
    labels = d['labels']

    # Compute per-class mean magnitude
    num_classes = int(labels.max()) + 1
    per_class_mag = np.zeros(num_classes)
    per_class_std = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_mag[c] = mags[mask].mean()
            per_class_std[c] = mags[mask].std()

    # Sort classes by magnitude (easy to hard, descending magnitude = easier)
    sorted_idx = np.argsort(per_class_mag)[::-1]
    sorted_mags = per_class_mag[sorted_idx]
    sorted_stds = per_class_std[sorted_idx]

    # Bar plot
    x_positions = np.arange(num_classes)
    style = STYLES[method]
    ax.bar(x_positions, sorted_mags, color=style['color'], alpha=0.7,
           edgecolor='none', width=1.0)
    ax.fill_between(x_positions, sorted_mags - sorted_stds, sorted_mags + sorted_stds,
                    color=style['color'], alpha=0.2)

    ax.set_xlabel('Classes (sorted by magnitude, descending)')
    ax.set_ylabel('Mean Magnitude')
    ax.set_title(f'{style["label"]}\nPer-class magnitude (final epoch)')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Annotate hardest/easiest
    hardest_class = sorted_idx[-1]
    easiest_class = sorted_idx[0]
    ax.annotate(f'Easiest: class {easiest_class}\n(mag={sorted_mags[0]:.3f})',
                xy=(0, sorted_mags[0]), xytext=(10, sorted_mags[0] - 0.15),
                fontsize=9, color='black')
    ax.annotate(f'Hardest: class {hardest_class}\n(mag={sorted_mags[-1]:.3f})',
                xy=(num_classes - 1, sorted_mags[-1]),
                xytext=(num_classes - 35, sorted_mags[-1] + 0.05),
                fontsize=9, color='black')

plt.tight_layout()
out_path = OUT_DIR / 'plot2_per_class_magnitude.png'
plt.savefig(out_path)
plt.close()
print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
# PLOT 3: cos(g_clean, g_aug) evolution — KEY METRIC
# ─────────────────────────────────────────────

print("\n[Plot 3] Augmentation Fidelity: cos(g_clean, g_aug) evolution...")
fig, ax = plt.subplots(figsize=(7, 4.5))

for method in METHODS:
    if method not in all_data:
        continue
    d = all_data[method]
    epochs = np.array(d['epochs'])
    cos_vals = np.array(d['cos_clean_aug_per_epoch'])

    # Smooth
    if len(cos_vals) >= args.smooth_window:
        cos_s = smooth(cos_vals, args.smooth_window)
        epochs_s = epochs[len(epochs) - len(cos_s):]
    else:
        cos_s = cos_vals
        epochs_s = epochs

    style = STYLES[method]
    ax.plot(epochs_s, cos_s,
            color=style['color'], linestyle=style['ls'],
            linewidth=style['lw'], label=style['label'])

ax.set_xlabel('Epoch')
ax.set_ylabel(r'$\cos(\nabla_z L_{\mathrm{clean}},\ \nabla_z L_{\mathrm{aug}})$')
ax.set_title('Augmentation Fidelity Across Training\n(Higher = augmented gradient aligned with clean direction)')
ax.legend(loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
out_path = OUT_DIR / 'plot3_cos_clean_aug.png'
plt.savefig(out_path)
plt.close()
print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
# PLOT 4: Training dynamics (loss + train_acc)
# ─────────────────────────────────────────────

print("\n[Plot 4] Training dynamics...")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax_loss, ax_acc = axes

for method in METHODS:
    if method not in all_data:
        continue
    d = all_data[method]
    epochs = np.array(d['epochs'])
    losses = np.array(d['train_loss'])
    accs = np.array(d['train_acc'])

    if len(losses) >= args.smooth_window:
        losses_s = smooth(losses, args.smooth_window)
        accs_s = smooth(accs, args.smooth_window)
        epochs_s = epochs[len(epochs) - len(losses_s):]
    else:
        losses_s, accs_s, epochs_s = losses, accs, epochs

    style = STYLES[method]
    ax_loss.plot(epochs_s, losses_s,
                 color=style['color'], linestyle=style['ls'],
                 linewidth=style['lw'], label=style['label'])
    ax_acc.plot(epochs_s, accs_s,
                color=style['color'], linestyle=style['ls'],
                linewidth=style['lw'], label=style['label'])

ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Training Loss')
ax_loss.set_title('Training Loss Evolution')
ax_loss.legend(loc='upper right', framealpha=0.95)
ax_loss.grid(True, alpha=0.3)
ax_loss.xaxis.set_major_locator(ticker.MultipleLocator(50))

ax_acc.set_xlabel('Epoch')
ax_acc.set_ylabel('Training Accuracy (%)')
ax_acc.set_title('Training Accuracy Evolution')
ax_acc.legend(loc='lower right', framealpha=0.95)
ax_acc.grid(True, alpha=0.3)
ax_acc.xaxis.set_major_locator(ticker.MultipleLocator(50))

plt.tight_layout()
out_path = OUT_DIR / 'plot4_training_dynamics.png'
plt.savefig(out_path)
plt.close()
print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
# Summary statistics for paper text
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY STATISTICS (for paper text)")
print("=" * 60)

print("\n📊 Cosine(g_clean, g_aug) — phase analysis:")
print(f"  {'Method':<25} {'Early (0-50)':>15} {'Mid (50-200)':>15} {'Late (200-300)':>15}")
print("  " + "-" * 75)
for method in METHODS:
    if method not in all_data:
        continue
    d = all_data[method]
    cos_vals = np.array(d['cos_clean_aug_per_epoch'])
    epochs = np.array(d['epochs'])

    early = cos_vals[(epochs >= 0) & (epochs < 50)]
    mid = cos_vals[(epochs >= 50) & (epochs < 200)]
    late = cos_vals[(epochs >= 200)]

    print(f"  {method:<25} {np.nanmean(early):>15.4f} {np.nanmean(mid):>15.4f} {np.nanmean(late):>15.4f}")

print("\n📊 Magnitude evolution:")
print(f"  {'Method':<25} {'Early (0-50)':>15} {'Mid (50-200)':>15} {'Late (200-300)':>15}")
print("  " + "-" * 75)
for method in METHODS:
    if method not in all_data:
        continue
    d = all_data[method]
    if d['magnitude_per_sample_final'] is None:
        print(f"  {method:<25} {'N/A (fixed)':>15} {'N/A (fixed)':>15} {'N/A (fixed)':>15}")
        continue
    mags = np.array(d['magnitude_mean'])
    epochs = np.array(d['epochs'])

    early = mags[(epochs >= 0) & (epochs < 50)]
    mid = mags[(epochs >= 50) & (epochs < 200)]
    late = mags[(epochs >= 200)]

    print(f"  {method:<25} {np.nanmean(early):>15.4f} {np.nanmean(mid):>15.4f} {np.nanmean(late):>15.4f}")

print("\n📊 Final test accuracy (best epoch, from CSV):")
print("  EntAugment_CutMix:    81.90%")
print("  RandAugment_CutMix:   81.21%")
print("  EntAugment_only:      79.39%")
print("  RandAugment_only:     78.17%")

print(f"\n✅ All plots saved to: {OUT_DIR}/")
print("\nFiles:")
for f in sorted(OUT_DIR.glob('*.png')):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.0f} KB)")