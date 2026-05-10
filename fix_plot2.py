"""
fix_plot2.py — Better visualization for per-class magnitude.

Improvements over original:
  - Annotations placed OUTSIDE bars (no overlap)
  - Add per-class top-5 easiest/hardest list
  - Show mean line for reference
  - Better color scheme matching plot style
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dump_dir', type=str, default='logs_dump')
parser.add_argument('--output', type=str, default='plots/plot2_per_class_magnitude_fixed.png')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

DUMP_DIR = Path(args.dump_dir)
OUT_PATH = Path(args.output)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# CIFAR-100 class names (for annotation)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

METHODS = {
    'EntAugment_CutMix': {'color': '#D62728', 'title': 'EntAugment + CutMix (Ours)'},
    'EntAugment_only':   {'color': '#FF7F0E', 'title': 'EntAugment only'},
}


def load_final_epoch(method, seed):
    folder = DUMP_DIR / f"{method}_seed{seed}"
    pkl_files = sorted(folder.glob('epoch_*.pkl'))
    if not pkl_files:
        return None
    with open(pkl_files[-1], 'rb') as f:
        d = pickle.load(f)
    return d


# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

for ax, (method, cfg) in zip(axes, METHODS.items()):
    d = load_final_epoch(method, args.seed)
    if d is None or d['magnitude_per_sample'] is None:
        continue

    mags = d['magnitude_per_sample']
    labels = d['labels']
    num_classes = int(labels.max()) + 1

    # Per-class statistics
    per_class_mag = np.zeros(num_classes)
    per_class_std = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_mag[c] = mags[mask].mean()
            per_class_std[c] = mags[mask].std()

    # Sort: easy → hard (descending magnitude)
    sorted_idx = np.argsort(per_class_mag)[::-1]
    sorted_mags = per_class_mag[sorted_idx]
    sorted_stds = per_class_std[sorted_idx]

    x = np.arange(num_classes)
    color = cfg['color']

    # Bar plot with std error band
    ax.bar(x, sorted_mags, color=color, alpha=0.85,
           edgecolor='none', width=1.0, label='Mean magnitude')
    ax.fill_between(x, sorted_mags - sorted_stds, sorted_mags + sorted_stds,
                    color=color, alpha=0.2, label='±1 std')

    # Reference: overall mean
    overall_mean = sorted_mags.mean()
    ax.axhline(y=overall_mean, color='black', linestyle='--', linewidth=1.2,
               alpha=0.6, label=f'Mean = {overall_mean:.3f}')

    # Top-3 easiest (left) and top-3 hardest (right)
    easiest_indices = sorted_idx[:3]
    hardest_indices = sorted_idx[-3:][::-1]  # reverse so highest entropy first

    # Build text boxes for annotations
    easy_text = "Easiest 3 classes:\n" + "\n".join([
        f"  {CIFAR100_CLASSES[c]} ({per_class_mag[c]:.3f})"
        for c in easiest_indices
    ])
    hard_text = "Hardest 3 classes:\n" + "\n".join([
        f"  {CIFAR100_CLASSES[c]} ({per_class_mag[c]:.3f})"
        for c in hardest_indices
    ])

    # Annotation in upper-right corner (no overlap with bars)
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.95, linewidth=0.8)

    ax.text(0.02, 0.98, easy_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top', horizontalalignment='left',
            bbox=bbox_props, family='monospace')

    ax.text(0.98, 0.98, hard_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top', horizontalalignment='right',
            bbox=bbox_props, family='monospace')

    # Compute spread metric
    spread = sorted_mags[0] - sorted_mags[-1]

    ax.set_xlabel('Classes (sorted by magnitude, descending)')
    ax.set_ylabel('Mean Magnitude')
    ax.set_title(f"{cfg['title']}\nSpread = {spread:.3f} | Mean = {overall_mean:.3f}")
    ax.set_ylim([0, 1.15])
    ax.set_xlim([-1, num_classes])
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    'Per-Class Magnitude Distribution at Final Epoch (CIFAR-100, ResNet-18)',
    fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig(OUT_PATH)
plt.close()
print(f"Saved: {OUT_PATH}")


# Print top-10 easy/hard for both methods
print("\n" + "=" * 70)
print("DETAILED CLASS RANKING (top-10 each)")
print("=" * 70)

for method, cfg in METHODS.items():
    d = load_final_epoch(method, args.seed)
    if d is None or d['magnitude_per_sample'] is None:
        continue

    mags = d['magnitude_per_sample']
    labels = d['labels']
    num_classes = int(labels.max()) + 1

    per_class_mag = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_mag[c] = mags[mask].mean()

    sorted_idx = np.argsort(per_class_mag)[::-1]

    print(f"\n{cfg['title']}:")
    print(f"  Top 10 EASIEST (high magnitude = high model confidence):")
    for i, c in enumerate(sorted_idx[:10]):
        print(f"    {i+1:2}. {CIFAR100_CLASSES[c]:<20} magnitude = {per_class_mag[c]:.4f}")
    print(f"  Top 10 HARDEST (low magnitude = low model confidence):")
    for i, c in enumerate(sorted_idx[-10:][::-1]):
        print(f"    {i+1:2}. {CIFAR100_CLASSES[c]:<20} magnitude = {per_class_mag[c]:.4f}")

print("\nDone.")