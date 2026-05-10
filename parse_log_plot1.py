"""
parse_log_plot1.py
Plot 1: Magnitude evolution over training epochs.
Parse từ log files (.txt) đã có sẵn — chạy được ngay không cần chạy lại.

Usage:
    python parse_log_plot1.py --log_dir logs/ --output plots/plot1_magnitude.png
"""
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='logs',
                    help='Directory chứa .log files')
parser.add_argument('--output', type=str, default='plots/plot1_magnitude.png')
parser.add_argument('--methods', type=str, nargs='+',
                    default=['cutmix', 'entcutmix', 'entaug', 'randaug'],
                    help='Prefix của log files cần parse')
parser.add_argument('--arch', type=str, default='r18',
                    choices=['r18', 'r50', 'wrn'])
parser.add_argument('--smooth', type=int, default=5,
                    help='Epoch smoothing window')
args = parser.parse_args()

LOG_DIR = Path(args.log_dir)
OUT_PATH = Path(args.output)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Regex patterns
RE_MAG  = re.compile(r'Mag:\s*([\d.]+)')
RE_LOSS = re.compile(r'Loss:\s*([\d.]+)')
RE_ACC  = re.compile(r'ACC:([\d.]+)')
RE_EPOCH = re.compile(r'Train Epoch:\s*(\d+)')


def smooth(arr, w):
    if w <= 1 or len(arr) < w:
        return np.array(arr)
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='valid')


def parse_log(path):
    """Parse một log file, trả về dict of lists."""
    epochs_mag = {}   # epoch → list of batch mag values
    test_acc   = {}   # epoch → best acc

    current_epoch = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Detect epoch
            em = RE_EPOCH.search(line)
            if em:
                current_epoch = int(em.group(1))
                if current_epoch not in epochs_mag:
                    epochs_mag[current_epoch] = []

            # Detect magnitude
            mm = RE_MAG.search(line)
            if mm and current_epoch is not None:
                epochs_mag[current_epoch].append(float(mm.group(1)))

            # Detect test acc
            am = RE_ACC.search(line)
            if am and current_epoch is not None:
                test_acc[current_epoch] = float(am.group(1))

    # Compute per-epoch mean magnitude
    epoch_list = sorted(epochs_mag.keys())
    mag_per_epoch = [np.mean(epochs_mag[e]) if epochs_mag[e] else np.nan
                     for e in epoch_list]
    acc_per_epoch = [test_acc.get(e, np.nan) for e in epoch_list]

    return {
        'epochs': epoch_list,
        'mag': mag_per_epoch,
        'acc': acc_per_epoch,
    }


# ── Collect data ──
METHOD_CONFIG = {
    'cutmix':    {'label': 'EntAugment + CutMix (Ours)', 'color': '#2196F3', 'ls': '-'},
    'entcutmix': {'label': 'EntAugment + EntCutMix',     'color': '#4CAF50', 'ls': '--'},
    'entaug':    {'label': 'EntAugment only',             'color': '#FF9800', 'ls': '-.'},
    'randaug':   {'label': 'RandAugment + CutMix',        'color': '#9C27B0', 'ls': ':'},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax_mag, ax_acc = axes

found_any = False
for method_prefix in args.methods:
    cfg_m = METHOD_CONFIG.get(method_prefix, {
        'label': method_prefix, 'color': 'gray', 'ls': '-'
    })

    # Find matching log files (3 seeds → average)
    pattern = f"{method_prefix}_c100_{args.arch}_s*.log"
    log_files = sorted(LOG_DIR.glob(pattern))

    if not log_files:
        print(f"⚠️ Không tìm thấy: {pattern}")
        continue

    print(f"Found {len(log_files)} files for {method_prefix}: {[f.name for f in log_files]}")

    all_mags = []
    all_accs = []
    epochs   = None

    for lf in log_files:
        parsed = parse_log(lf)
        if not parsed['epochs']:
            print(f"  ⚠️ Empty: {lf.name}")
            continue
        all_mags.append(parsed['mag'])
        all_accs.append(parsed['acc'])
        if epochs is None:
            epochs = parsed['epochs']

    if not all_mags:
        continue

    found_any = True
    min_len = min(len(m) for m in all_mags)
    epochs = epochs[:min_len]

    # Mean ± std across seeds
    mag_arr = np.array([m[:min_len] for m in all_mags])
    acc_arr = np.array([a[:min_len] for a in all_accs])

    mag_mean = np.nanmean(mag_arr, axis=0)
    mag_std  = np.nanstd(mag_arr, axis=0)
    acc_mean = np.nanmean(acc_arr, axis=0)

    # Smooth
    mag_s = smooth(mag_mean, args.smooth)
    acc_s = smooth(acc_mean, args.smooth)
    ep_s  = epochs[len(epochs) - len(mag_s):]

    ax_mag.plot(ep_s, mag_s,
                label=cfg_m['label'],
                color=cfg_m['color'],
                linestyle=cfg_m['ls'],
                linewidth=1.8)
    if len(log_files) > 1:
        std_s = smooth(mag_std, args.smooth)
        ax_mag.fill_between(ep_s,
                            mag_s - std_s, mag_s + std_s,
                            alpha=0.15, color=cfg_m['color'])

    ax_acc.plot(ep_s, acc_s,
                label=cfg_m['label'],
                color=cfg_m['color'],
                linestyle=cfg_m['ls'],
                linewidth=1.8)

if not found_any:
    print("❌ Không tìm thấy log nào. Kiểm tra --log_dir và --methods.")
    exit(1)

# ── Formatting ──
ax_mag.set_xlabel('Epoch', fontsize=12)
ax_mag.set_ylabel('Augmentation Magnitude (mean)', fontsize=12)
ax_mag.set_title('Entropy-Driven Magnitude Evolution', fontsize=13)
ax_mag.legend(fontsize=9)
ax_mag.grid(True, alpha=0.3)
ax_mag.xaxis.set_major_locator(ticker.MultipleLocator(20))

ax_acc.set_xlabel('Epoch', fontsize=12)
ax_acc.set_ylabel('Test Accuracy (%)', fontsize=12)
ax_acc.set_title('Test Accuracy over Training', fontsize=13)
ax_acc.legend(fontsize=9)
ax_acc.grid(True, alpha=0.3)
ax_acc.xaxis.set_major_locator(ticker.MultipleLocator(20))

plt.suptitle(f'CIFAR-100 / {args.arch.upper()} — Training Dynamics', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {OUT_PATH}")
plt.show()