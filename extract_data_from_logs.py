"""
extract_data_from_logs.py
Parse log files đã có → xuất .npy files cho plotting.

Usage:
    python extract_data_from_logs.py --log_dir logs/ --arch wrn --dataset c100
"""
import re
import argparse
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--log_dir', type=str, default='logs')
ap.add_argument('--arch', type=str, default='wrn', choices=['r18', 'r50', 'wrn'])
ap.add_argument('--dataset', type=str, default='c100', choices=['c10', 'c100'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--out_dir', type=str, default='figures/data')
args = ap.parse_args()

LOG_DIR = Path(args.log_dir)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RE_ACC = re.compile(r'ACC:([\d.]+)')
RE_MAG = re.compile(r'Mag:\s*([\d.]+)')
RE_EPOCH_LINE = re.compile(r'Train Epoch:\s*(\d+)')

# Method → log file prefix mapping
METHOD_LOG = {
    'RandAugment':           f'randaug_{args.dataset}_{args.arch}_s{args.seed}',
    'EntAugment':            f'entaug_{args.dataset}_{args.arch}_s{args.seed}',
    'CutMix':                f'purecutmix_{args.dataset}_{args.arch}_s{args.seed}',
    'RA+CutMix':             f'racutmix_{args.dataset}_{args.arch}_s{args.seed}',
    'EntAug+EntCutMix':      f'entcutmix_{args.dataset}_{args.arch}_s{args.seed}',
    'EntAugment-CutMix':     f'cutmix_{args.dataset}_{args.arch}_s{args.seed}',
}

# Also check logging-style names
METHOD_LOG_ALT = {
    'EntAugment':        f'logging_EntAugment_only_CIFAR100_{args.arch}_s{args.seed}',
    'EntAugment-CutMix': f'logging_EntAugment_CutMix_CIFAR100_{args.arch}_s{args.seed}',
}


def parse_log(path):
    """Parse 1 log file → acc_per_epoch, mag_per_epoch."""
    acc_per_epoch  = {}
    mag_sum        = {}
    mag_count      = {}
    current_epoch  = None

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            em = RE_EPOCH_LINE.search(line)
            if em:
                current_epoch = int(em.group(1))
                if current_epoch not in mag_sum:
                    mag_sum[current_epoch]   = 0.0
                    mag_count[current_epoch] = 0

            mm = RE_MAG.search(line)
            if mm and current_epoch is not None:
                mag_sum[current_epoch]   += float(mm.group(1))
                mag_count[current_epoch] += 1

            am = RE_ACC.search(line)
            if am and current_epoch is not None:
                acc_per_epoch[current_epoch] = float(am.group(1))

    epochs = sorted(acc_per_epoch.keys())
    acc_arr = np.array([acc_per_epoch[e] for e in epochs])
    mag_arr = np.array([
        mag_sum[e] / mag_count[e] if mag_count.get(e, 0) > 0 else np.nan
        for e in epochs
    ])
    return acc_arr, mag_arr, epochs


def find_log(prefix):
    """Find log file by prefix."""
    candidates = list(LOG_DIR.glob(f"{prefix}.log")) + \
                 list(LOG_DIR.glob(f"{prefix}*.log"))
    if candidates:
        return candidates[0]
    return None


print(f"Extracting from: {LOG_DIR}/  →  {OUT_DIR}/")
print(f"Arch={args.arch}, Dataset={args.dataset}, Seed={args.seed}\n")

found_any = False
for method, prefix in METHOD_LOG.items():
    log_path = find_log(prefix)

    # Fallback to alt naming
    if log_path is None and method in METHOD_LOG_ALT:
        log_path = find_log(METHOD_LOG_ALT[method])

    if log_path is None:
        print(f"  ⚠️  NOT FOUND: {prefix}.log")
        continue

    print(f"  Parsing: {log_path.name}")
    try:
        acc, mag, epochs = parse_log(log_path)
        safe_name = method.replace('+', 'plus').replace(' ', '_').replace('-', '_')

        np.save(OUT_DIR / f'acc_{safe_name}.npy', acc)
        np.save(OUT_DIR / f'mag_{safe_name}.npy', mag)
        np.save(OUT_DIR / f'epochs_{safe_name}.npy', np.array(epochs))

        print(f"    ✅ {len(epochs)} epochs | max_acc={acc.max():.2f}% | "
              f"mean_mag={np.nanmean(mag):.4f}")
        found_any = True
    except Exception as e:
        print(f"    ❌ Error: {e}")

if found_any:
    print(f"\n✅ Data saved to {OUT_DIR}/")
    print("Run plot_figures.py to generate figures.")
else:
    print("\n❌ No logs found. Check --log_dir and file naming.")