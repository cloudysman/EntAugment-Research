"""
train_with_logging.py — FINAL version

Extended logging cho 4 methods để vẽ plots cho paper:
  - Plot 1: Magnitude evolution (từ log txt — script riêng)
  - Plot 2: Per-class magnitude evolution (dump trainset.MAGNITUDE + labels mỗi epoch)
  - Plot 3a: cos(g_ema, g_batch) — gradient temporal stability
  - Plot 3b: cos(g_clean, g_aug) — augmentation fidelity (CRITICAL metric for paper story)

Methods supported (--method):
  - EntAugment_CutMix     (ours)
  - RandAugment_CutMix    (control: fixed-magnitude + spatial mixing)
  - EntAugment_only       (control: adaptive magnitude alone)
  - RandAugment_only      (control: fixed-magnitude alone)

Dump format: pickle, 1 file/epoch trong logs_dump/{method}_seed{seed}/epoch_XXX.pkl

Overhead vs original: ~3-5% (1 extra forward pass for clean batch every grad_log_interval)
"""
import os
import random
import argparse
import csv
import pickle
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--conf', default='./confs/resnet18.yaml', type=str)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--cutout_length', type=int, default=16)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--num_worker', type=int, default=8)
parser.add_argument('--aug', type=str, default='entaugment')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cutmix_prob', type=float, default=0.5)
parser.add_argument('--cutmix_alpha', type=float, default=1.0)
parser.add_argument('--method', type=str, required=True,
                    choices=['EntAugment_CutMix', 'RandAugment_CutMix',
                             'EntAugment_only', 'RandAugment_only'])
parser.add_argument('--result_file', type=str, default='benchmark_composition_results.csv')
parser.add_argument('--dump_dir', type=str, default='logs_dump')
parser.add_argument('--grad_log_interval', type=int, default=10,
                    help='Compute g_ema/g_clean alignment every N batches (default 10)')
parser.add_argument('--ra_n', type=int, default=2)
parser.add_argument('--ra_m', type=int, default=9)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from Dataset import CIFAR10Dataset, CIFAR100Dataset
from Network import *
from organize_transform import make_transform
from augmentation import trivialaugment
from augmentation.entaugment import ALL_TRANSFORMS as ENT_OPS
from warmup_scheduler import GradualWarmupScheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# ── Method config ──
USE_CUTMIX = 'CutMix' in args.method
USE_ENTAUG = 'EntAugment' in args.method
USE_RANDAUG = 'RandAugment' in args.method

print(f"Method: {args.method}")
print(f"  USE_CUTMIX={USE_CUTMIX}, USE_ENTAUG={USE_ENTAUG}, USE_RANDAUG={USE_RANDAUG}")

# ── Setup dump directory ──
dump_dir = Path(args.dump_dir) / f"{args.method}_seed{args.seed}"
dump_dir.mkdir(parents=True, exist_ok=True)
print(f"Dump dir: {dump_dir}")

with open(args.conf) as f:
    cfg = yaml.safe_load(f)

# ── RandAugment ──
trivialaugment.set_augmentation_space(augmentation_space='standard', num_strengths=30)


class RandAugment:
    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m

    def __call__(self, img):
        ops = random.choices(ENT_OPS, k=self.n)
        for op in ops:
            img = op.pil_transformer(1.0, self.m)(img)
        return img


# ── Transforms ──
NORM_MEAN = [x / 255.0 for x in [125.3, 123.0, 113.9]]
NORM_STD = [x / 255.0 for x in [63.0, 62.1, 66.7]]

# Clean transform: chỉ crop+flip+normalize (cho Plot 3b: g_clean reference)
transform_clean = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# RandAugment transform: thêm RandAugment vào transform_clean
transform_randaug = transforms.Compose([
    transforms.ToPILImage(),
    RandAugment(n=args.ra_n, m=args.ra_m),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# ── Model ──
NUM_CLASSES = num_class(args.dataset.lower())
best_acc = 0
best_epoch = 0
warmup_epoch = 10
acc_list = []

model = get_model(cfg['model']['type'], num_classes=NUM_CLASSES)
model = torch.nn.DataParallel(
    model, device_ids=np.arange(len(args.gpus.split(','))).tolist()
).cuda()

if cfg['optimizer']['type'] == 'sgd':
    optimizer = optim.SGD(
        model.parameters(), lr=cfg['lr'], momentum=args.momentum,
        weight_decay=cfg['optimizer']['decay'],
        nesterov=cfg['optimizer']['nesterov']
    )

lr_type = cfg['lr_schedule']['type']
if lr_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epoch'], eta_min=0.)
elif lr_type == 'step':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg['lr_schedule']['milestones'],
        gamma=cfg['lr_schedule']['gamma'])

if cfg['lr_schedule']['warmup'] != '' and cfg['lr_schedule']['warmup']['epoch'] > 0:
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=cfg['lr_schedule']['warmup']['multiplier'],
        total_epoch=cfg['lr_schedule']['warmup']['epoch'],
        after_scheduler=scheduler
    )

epoches = cfg['epoch']
criterion = nn.CrossEntropyLoss(reduction='none')

# ── Dataset ──
if args.dataset == 'CIFAR10':
    root = 'data/CIFAR10/'
    trainset = CIFAR10Dataset(root=root, train=True, transform=transform_clean, aug='entaugment')
    testset = CIFAR10Dataset(root=root, train=False, transform=transform_test)
elif args.dataset == 'CIFAR100':
    root = 'data/CIFAR100/cifar-100-python/'
    trainset = CIFAR100Dataset(root, train=True, fine_label=True,
                               transform=transform_clean, aug='entaugment')
    testset = CIFAR100Dataset(root, train=False, fine_label=True,
                              transform=transform_test)

# Configure trainset based on method
if USE_RANDAUG:
    # RandAugment methods use external_transform (bypass EntAugment logic)
    trainset.external_transform = transform_randaug
elif USE_ENTAUG:
    # EntAugment methods use Dataset's built-in EntAugment logic
    pass  # default behavior
else:
    raise ValueError(f"Method {args.method} không có augmentation pipeline.")

train_loader = DataLoader(trainset, batch_size=cfg['batch'],
                          shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(testset, batch_size=cfg['batch'],
                         shuffle=False, num_workers=8, pin_memory=True)

start_epoch = 0
if args.resume:
    checkpoint = torch.load('./checkpoint/{}/{}'.format(args.dataset, args.resume))
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# ─────────────────────────────────────────────
# Helper: get classifier layer (handles ResNet/WRN/ShakeResNet)
# ─────────────────────────────────────────────

def get_classifier_layer(net):
    m = net.module if hasattr(net, 'module') else net
    for name in ['linear', 'fc_out', 'fc', 'classifier']:
        if hasattr(m, name):
            layer = getattr(m, name)
            if isinstance(layer, nn.Linear):
                return layer
    raise ValueError("Cannot find classifier layer")

classifier = get_classifier_layer(model)
print(f"Classifier: weight shape = {classifier.weight.shape}")


# ─────────────────────────────────────────────
# Analytical gradient: g = W^T @ (p - y)
# ─────────────────────────────────────────────

def analytical_grad(p, labels, W, num_classes):
    """
    Compute ∇_z L (gradient w.r.t. features) analytically.
    
    Args:
        p: softmax probabilities [B, K]
        labels: class indices [B]
        W: classifier weight [K, D]
        num_classes: K
    Returns:
        g: gradient [B, D] (per-sample)
    """
    y_onehot = F.one_hot(labels, num_classes).float()
    return torch.matmul(p - y_onehot, W.detach())


# ─────────────────────────────────────────────
# CutMix helpers
# ─────────────────────────────────────────────

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(inputs, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0)).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
    return inputs, labels, labels[index], lam


# ─────────────────────────────────────────────
# Helper: get clean version of a batch
# ─────────────────────────────────────────────

def get_clean_batch(idx_tensor):
    """
    Get clean version (only crop+flip+normalize) of samples at idx.
    Used for computing g_clean for Plot 3b.
    """
    clean_imgs = []
    if hasattr(trainset, 'data'):
        if hasattr(trainset, 'targets'):  # CIFAR10
            data_array = trainset.data
        else:  # CIFAR100
            data_array = trainset.data
    
    for i in idx_tensor.cpu().numpy():
        raw_img = data_array[i]  # numpy [32, 32, 3]
        clean_imgs.append(transform_clean(raw_img))
    
    return torch.stack(clean_imgs).cuda()


# ─────────────────────────────────────────────
# Training loop with extended logging
# ─────────────────────────────────────────────

def train(net, epoch):
    global optimizer
    net.train()

    training_loss = 0.0
    training_magnitude = 0.0
    total = len(train_loader.dataset)
    correct = 0

    # ── Logging buffers (RAM, dump 1 lần/epoch) ──
    cos_ema_history = []      # Plot 3a: cos(g_ema, g_batch)
    cos_clean_aug_history = []  # Plot 3b: cos(g_clean, g_aug)
    batch_mag_mean = []
    g_ema = None
    EMA_ALPHA = 0.1

    W_classifier = classifier.weight  # [K, D]

    for i, data in enumerate(train_loader, 0):
        idx, inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        use_cutmix_this_batch = USE_CUTMIX and (np.random.random() < args.cutmix_prob)

        # ── Forward (augmented) ──
        optimizer.zero_grad()
        if use_cutmix_this_batch:
            inputs_mixed, labels_a, labels_b, lam = cutmix_data(
                inputs.clone(), labels, alpha=args.cutmix_alpha
            )
            outputs = net(inputs_mixed)
            loss = lam * criterion(outputs, labels_a).mean() \
                 + (1 - lam) * criterion(outputs, labels_b).mean()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, labels).mean()

        # ── EntAugment magnitude (only for EntAugment methods) ──
        if USE_ENTAUG:
            with torch.no_grad():
                probability = F.softmax(outputs, dim=1)
                entropy_val = -torch.sum(probability * torch.log(probability + 1e-8), dim=1)
                entropy_norm = entropy_val / np.log(NUM_CLASSES)
                magnitude = (1.0 - entropy_norm).clamp(0.0, 1.0)
                trainset.set_MAGNITUDE(idx, magnitude.detach().cpu())
                training_magnitude += magnitude.mean().item()
                batch_mag_mean.append(magnitude.mean().item())

        training_loss += loss.item()
        _, predicted = outputs.max(1)

        # ── Gradient logging (every grad_log_interval batches) ──
        if (i + 1) % args.grad_log_interval == 0:
            with torch.no_grad():
                # Plot 3b: cos(g_clean, g_aug) — REQUIRES extra forward pass on clean
                # Get clean version of current batch
                try:
                    clean_inputs = get_clean_batch(idx)
                    
                    # Forward clean
                    outputs_clean = net(clean_inputs)
                    p_clean = F.softmax(outputs_clean, dim=1)
                    g_clean = analytical_grad(p_clean, labels, W_classifier, NUM_CLASSES)  # [B, D]
                    g_clean_mean = g_clean.mean(dim=0)  # [D]
                    
                    # g_aug from current outputs (already computed)
                    p_aug = F.softmax(outputs.detach(), dim=1)
                    if use_cutmix_this_batch:
                        # For CutMix: target is mixed label, use lambda-weighted
                        y_onehot_a = F.one_hot(labels_a, NUM_CLASSES).float()
                        y_onehot_b = F.one_hot(labels_b, NUM_CLASSES).float()
                        y_mixed = lam * y_onehot_a + (1 - lam) * y_onehot_b
                        g_aug = torch.matmul(p_aug - y_mixed, W_classifier.detach())
                    else:
                        g_aug = analytical_grad(p_aug, labels, W_classifier, NUM_CLASSES)
                    g_aug_mean = g_aug.mean(dim=0)  # [D]
                    
                    # Cosine similarity (Plot 3b)
                    cos_clean_aug = F.cosine_similarity(
                        g_clean_mean.unsqueeze(0),
                        g_aug_mean.unsqueeze(0)
                    ).item()
                    cos_clean_aug_history.append(cos_clean_aug)
                    
                    # EMA tracking (Plot 3a) — use g_aug_mean
                    if g_ema is None:
                        g_ema = g_aug_mean.clone()
                        cos_ema_history.append(1.0)  # first batch
                    else:
                        cos_ema = F.cosine_similarity(
                            g_ema.unsqueeze(0),
                            g_aug_mean.unsqueeze(0)
                        ).item()
                        cos_ema_history.append(cos_ema)
                        g_ema = EMA_ALPHA * g_aug_mean + (1 - EMA_ALPHA) * g_ema
                
                except Exception as e:
                    print(f"[Warning] Gradient logging failed at batch {i}: {e}")

        # ── Backward + step ──
        loss.backward()
        optimizer.step()

        if use_cutmix_this_batch:
            correct += (lam * predicted.eq(labels_a).sum().float()
                      + (1 - lam) * predicted.eq(labels_b).sum().float()).item()
        else:
            correct += predicted.eq(labels).sum().item()

        if (i + 1) % args.log_interval == 0:
            trained_total = (i + 1) * len(inputs)
            mag_str = f" Mag: {training_magnitude/(i+1):.4f}" if USE_ENTAUG else ""
            cos_str = ""
            if cos_clean_aug_history:
                cos_str = f" CosCleanAug: {np.mean(cos_clean_aug_history[-10:]):.4f}"
            print('Train Epoch: {} [{}] [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.4f}{}{} Acc: {:.2f}'.format(
                epoch, args.method,
                trained_total, total, 100. * trained_total / total,
                training_loss / (i + 1), mag_str, cos_str,
                100. * correct / trained_total))

    if epoch >= warmup_epoch:
        trainset.is_warmup_finished = True

    # ── Dump epoch log (RAM → disk, 1 write/epoch) ──
    epoch_log = {
        'epoch': epoch,
        'method': args.method,
        'seed': args.seed,
        # Plot 2 data
        'magnitude_per_sample': trainset.MAGNITUDE.numpy().copy() if USE_ENTAUG else None,
        'labels': np.array(trainset.labels) if hasattr(trainset, 'labels') else np.array(trainset.targets),
        # Plot 3a, 3b data
        'cos_g_ema_batch': np.array(cos_ema_history),
        'cos_g_clean_aug': np.array(cos_clean_aug_history),
        # Training metrics
        'batch_mag_mean': np.array(batch_mag_mean),
        'train_loss': training_loss / len(train_loader),
        'train_acc': 100. * correct / total,
    }

    dump_path = dump_dir / f"epoch_{epoch:03d}.pkl"
    with open(dump_path, 'wb') as f:
        pickle.dump(epoch_log, f)


def test(net, epoch):
    global best_acc, best_epoch
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct * 100. / total
    print('EPOCH:{}, ======================ACC:{}===================='.format(epoch, acc))
    acc_list.append(acc)
    if acc >= best_acc:
        best_acc = acc
        best_epoch = epoch
    print('BEST EPOCH:{},BEST ACC:{}'.format(best_epoch, best_acc))


if __name__ == '__main__':
    print(f'=== {args.method} with Extended Logging ===')
    print(f'Seed: {args.seed}, Dataset: {args.dataset}, Model: {cfg["model"]["type"]}, Epochs: {epoches}')
    print(f'Grad log interval: every {args.grad_log_interval} batches')
    print(f'Dump dir: {dump_dir}')

    for epoch in tqdm(range(start_epoch, epoches)):
        train(model, epoch)
        test(model, epoch)
        scheduler.step()

    # Save summary
    result_file = args.result_file
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['method', 'dataset', 'model', 'seed', 'best_epoch', 'best_acc'])
        writer.writerow([args.method, args.dataset, cfg['model']['type'],
                         args.seed, best_epoch, best_acc])
    print(f'Result saved to {result_file}')
    print(f'Epoch logs saved to {dump_dir}/')