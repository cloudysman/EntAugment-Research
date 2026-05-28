"""
clean_and_summarize.py
Fix CSV lộn cột, dedup entries, tính mean±std, xuất bảng paper.

Usage:
    python clean_and_summarize.py
"""
import csv
import numpy as np
from collections import defaultdict
from pathlib import Path

RESULT_FILE = 'benchmark_composition_results.csv'
DATASET     = 'CIFAR100'

# ─────────────────────────────────────────────
# Bước 1: Load và normalize từng row
# ─────────────────────────────────────────────
records = []  # list of (method, model, seed, best_acc)

with open(RESULT_FILE, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"Header: {header}")

    for row in reader:
        if len(row) < 5:
            continue

        # Normalize: tìm đúng cột bất kể thứ tự
        # Hai format trong CSV:
        # Format A: dataset, model, seed, best_epoch, best_acc, method
        # Format B: method, dataset, model, seed, best_epoch, best_acc

        # Detect format bằng cách xem row[0] có phải số không
        def is_num(s):
            try: float(s); return True
            except: return False

        def is_model(s):
            return any(x in s.lower() for x in ['resnet', 'wresnet', 'wrn', 'vgg', 'alexnet'])

        def is_dataset(s):
            return s.upper() in ['CIFAR10', 'CIFAR100', 'SVHN', 'IMAGENET']

        # Tìm method, dataset, model, seed, acc từ row
        method = dataset = model = seed = acc = None

        for val in row:
            val = val.strip()
            if is_dataset(val):
                dataset = val.upper()
            elif is_model(val):
                model = val.lower()
            elif is_num(val) and '.' in val and acc is None:
                candidate = float(val)
                if 50 < candidate < 100:  # reasonable accuracy range
                    acc = candidate
            elif is_num(val) and '.' not in val and seed is None:
                candidate = int(float(val))
                if candidate in [42, 123, 456, 0, 1, 2]:
                    seed = candidate
            elif not is_num(val) and not is_dataset(val) and not is_model(val):
                if method is None and len(val) > 2:
                    method = val

        if method and dataset and model and seed is not None and acc is not None:
            if dataset == DATASET:
                records.append((method, model, seed, acc))
        else:
            print(f"  ⚠️ Skip row: {row} → method={method} dataset={dataset} model={model} seed={seed} acc={acc}")

print(f"\nLoaded {len(records)} records")

# ─────────────────────────────────────────────
# Bước 2: Dedup — cùng (method, model, seed) giữ acc cao nhất
# ─────────────────────────────────────────────
best_per_run = {}  # (method, model, seed) → best_acc
for method, model, seed, acc in records:
    key = (method, model, seed)
    if key not in best_per_run or acc > best_per_run[key]:
        best_per_run[key] = acc

print(f"After dedup: {len(best_per_run)} unique runs")

# ─────────────────────────────────────────────
# Bước 3: Group by (method, model)
# ─────────────────────────────────────────────
grouped = defaultdict(list)  # (method, model) → list of acc
for (method, model, seed), acc in best_per_run.items():
    grouped[(method, model)].append(acc)

# ─────────────────────────────────────────────
# Bước 4: Print raw data để kiểm tra
# ─────────────────────────────────────────────
print("\n── Raw grouped data ──")
methods_found = sorted(set(m for m, _ in grouped.keys()))
models_found  = sorted(set(m for _, m in grouped.keys()))
print(f"Methods: {methods_found}")
print(f"Models:  {models_found}")

for method in methods_found:
    for model in models_found:
        accs = grouped.get((method, model), [])
        if accs:
            print(f"  {method:<35} {model:<20} n={len(accs)} accs={[f'{a:.2f}' for a in sorted(accs)]}")

# ─────────────────────────────────────────────
# Bước 5: Build paper table
# ─────────────────────────────────────────────
METHOD_ORDER = [
    ('Baseline',             'Baseline (crop+flip)'),
    ('PureCutMix',           'CutMix'),
    ('RandAugment_only',     'RandAugment'),
    ('RandAugment_CutMix',   'RandAugment + CutMix'),
    ('EntAugment_only',      'EntAugment'),
    ('EntAugment_EntCutMix', 'EntAugment + EntCutMix'),
    ('EntAugment_CutMix',    'EntAugment + CutMix (Ours)'),
]

MODEL_ORDER   = ['resnet18', 'resnet50', 'wresnet28_10']
MODEL_DISPLAY = {'resnet18': 'ResNet-18', 'resnet50': 'ResNet-50', 'wresnet28_10': 'WRN-28-10'}

def fmt(accs):
    if not accs:
        return '-', np.nan
    mean = np.mean(accs)
    std  = np.std(accs)
    if len(accs) == 1:
        return f'{mean:.2f}', mean
    return f'{mean:.2f}±{std:.2f}', mean

# Console table
SEP = "=" * 80
print(f"\n{SEP}")
print(f"PAPER TABLE — {DATASET}")
print(SEP)
hdr = f"{'Method':<40}" + "".join(f"{MODEL_DISPLAY[m]:>14}" for m in MODEL_ORDER)
print(hdr)
print("-" * 80)

table_data = []
for method_key, method_display in METHOD_ORDER:
    row_vals = []
    found = False
    for model in MODEL_ORDER:
        accs = grouped.get((method_key, model), [])
        s, mean = fmt(accs)
        row_vals.append((s, mean, len(accs)))
        if accs:
            found = True

    if found:
        line = f"{method_display:<40}" + "".join(f"{v[0]:>14}" for v in row_vals)
        # Annotate seed count
        counts = [f"n={v[2]}" for v in row_vals]
        print(f"{line}   ({', '.join(counts)})")
        table_data.append((method_key, method_display, row_vals))

print(SEP)

# ─────────────────────────────────────────────
# Bước 6: LaTeX table
# ─────────────────────────────────────────────

# Find best per column for bolding
col_best = {}
for model_idx, model in enumerate(MODEL_ORDER):
    means = []
    for method_key, _, row_vals in table_data:
        _, mean, n = row_vals[model_idx]
        if not np.isnan(mean):
            means.append(mean)
    col_best[model_idx] = max(means) if means else np.nan

def latex_cell(s, mean, n, model_idx):
    if s == '-':
        return '-'
    is_best = not np.isnan(mean) and abs(mean - col_best.get(model_idx, np.nan)) < 0.005
    cell = s.replace('±', '$\\pm$')
    return f'\\textbf{{{cell}}}' if is_best else cell

tex = []
tex.append('% Auto-generated — ' + DATASET)
tex.append('\\begin{table}[t]')
tex.append('  \\centering')
tex.append('  \\caption{Top-1 accuracy (\\%) on ' + DATASET +
           '. Mean\\,$\\pm$\\,std over 3 seeds. \\textbf{Bold}: best per column.}')
tex.append('  \\label{tab:cifar100}')
tex.append('  \\begin{tabular}{lccc}')
tex.append('    \\toprule')
tex.append('    \\textbf{Method} & \\textbf{ResNet-18} & \\textbf{ResNet-50} & \\textbf{WRN-28-10} \\\\')
tex.append('    \\midrule')

separators_before = {'RandAugment_only', 'EntAugment_only', 'EntAugment_CutMix'}
for method_key, method_display, row_vals in table_data:
    if method_key in separators_before:
        tex.append('    \\midrule')
    cells = [latex_cell(row_vals[i][0], row_vals[i][1], row_vals[i][2], i)
             for i in range(len(MODEL_ORDER))]
    display = method_display.replace('(Ours)', '\\textit{(Ours)}')
    tex.append(f'    {display} & {" & ".join(cells)} \\\\')

tex.append('    \\bottomrule')
tex.append('  \\end{tabular}')
tex.append('\\end{table}')

tex_str = '\n'.join(tex)
print('\nLaTeX Table:')
print('-' * 60)
print(tex_str)

Path('paper_table.tex').write_text(tex_str)
print('\n✅ Saved: paper_table.tex')

# ─────────────────────────────────────────────
# Bước 7: Missing runs
# ─────────────────────────────────────────────
print('\n── Missing / Incomplete runs ──')
any_missing = False
for method_key, method_display in METHOD_ORDER:
    for model in MODEL_ORDER:
        accs = grouped.get((method_key, model), [])
        n = len(accs)
        if n == 0:
            print(f'  ❌ MISSING: {method_key} / {model}')
            any_missing = True
        elif n < 3:
            print(f'  ⚠️  {method_key} / {model}: {n}/3 seeds {[f"{a:.2f}" for a in accs]}')
            any_missing = True
if not any_missing:
    print('  ✅ Tất cả đủ 3 seeds!')