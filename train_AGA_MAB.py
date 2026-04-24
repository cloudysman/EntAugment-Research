"""
AGA-MAB: Analytical Gradient-Aligned Multi-Armed Bandit
for Adaptive Data Augmentation

Based on EntAugment + per-batch MAB, upgraded with:
- Gradient Alignment as reward signal (replaces noisy val_loss reward)
- Analytical gradient computation (zero backward overhead)
- Sliding-window UCB (handles non-stationary training dynamics)
- Per-sample cosine similarity (theoretically grounded via IB)

Key differences from train_EntAugment_MAB_perBatch.py:
1. MAB reward = cosine(∇_z L_clean, ∇_z L_aug) instead of val_loss delta
2. No compute_val_loss() needed anymore
3. Sliding window resets each epoch
4. Clean forward pass (no_grad) for analytical gradient
"""

import pathlib
import sys
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument('--log_interval', type=int, default=50, help='log training status')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--conf', default='./confs/resnet18.yaml', type=str, help='yaml file')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--cutout_length', type=int, default=16)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--num_worker', type=int, default=8, choices=[2, 4, 8, 16, 32])
parser.add_argument('--aug', type=str, default='entaugment')
parser.add_argument('--ucb_c', type=float, default=1.0,
                    help='UCB exploration constant')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

import random
import numpy as np
import math
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
import csv
from Dataset import CIFAR10Dataset, CIFAR100Dataset
from Network import *
from augmentation.entaugment import ALL_TRANSFORMS, PARAMETER_MAX
from augmentation.cutout import Cutout
from augmentation import trivialaugment
import yaml
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

with open(args.conf) as f:
    cfg = yaml.safe_load(f)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
])
transform_warmup = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
])

best_acc = 0
best_epoch = 0
warmup_epoch = 10
acc_list = []
NUM_CLASSES = num_class(args.dataset.lower())
model = get_model(cfg['model']['type'], num_classes=NUM_CLASSES)
model = torch.nn.DataParallel(
    model, device_ids=np.arange(len(args.gpus.split(','))).tolist()
).cuda()

if cfg['optimizer']['type'] == 'sgd':
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['lr'],
        momentum=args.momentum,
        weight_decay=cfg['optimizer']['decay'],
        nesterov=cfg['optimizer']['nesterov']
    )

lr_schduler_type = cfg['lr_schedule']['type']
if lr_schduler_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epoch'], eta_min=0.)
elif lr_schduler_type == 'step':
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

if args.dataset == 'CIFAR10':
    root = 'data/CIFAR10/'
    trainset = CIFAR10Dataset(root=root, train=True, transform=transform_warmup, aug=args.aug)
    testset = CIFAR10Dataset(root=root, train=False, transform=transform_test)
elif args.dataset == 'CIFAR100':
    root = 'data/CIFAR100/cifar-100-python/'
    trainset = CIFAR100Dataset(root, train=True, fine_label=True,
                               transform=transform_warmup, aug=args.aug)
    testset = CIFAR100Dataset(root, train=False, fine_label=True,
                              transform=transform_test)

train_loader = DataLoader(dataset=trainset, batch_size=cfg['batch'],
                          shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(dataset=testset, batch_size=cfg['batch'],
                         shuffle=False, num_workers=8, pin_memory=True)

start_epoch = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}/{}'.format(args.dataset, args.resume))
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# ─────────────────────────────────────────────
# [AGA-MAB NEW] Analytical Gradient-Aligned MAB
# ─────────────────────────────────────────────

def get_classifier_layer(net):
    """
    Extract the final Linear classifier layer from any supported architecture.
    Handles DataParallel wrapping and different naming conventions:
      - ResNet18/50, WideResNet: self.linear
      - ShakeResNet, ShakeResNeXt, ShakePyramidNet: self.fc_out
      - resnet2 (ResNet32/44/56/110): self.fc
      - PyramidNet: self.fc
    """
    m = net.module if hasattr(net, 'module') else net
    for name in ['linear', 'fc_out', 'fc', 'classifier']:
        if hasattr(m, name):
            layer = getattr(m, name)
            if isinstance(layer, nn.Linear):
                return layer
    raise ValueError(
        "Cannot find classifier layer. Available: {}".format(
            [n for n, _ in m.named_modules() if isinstance(_, nn.Linear)]
        )
    )


def compute_analytical_gradient(p, labels, classifier_weight, num_classes):
    """
    Compute ∇_z L = W^T (p - y) analytically.
    No backward pass needed — only matrix multiplication.

    Args:
        p: softmax probabilities, shape [B, K]
        labels: class indices, shape [B]
        classifier_weight: W from final Linear layer, shape [K, D]
        num_classes: K

    Returns:
        g: analytical gradient w.r.t. features z, shape [B, D]
    """
    y_onehot = F.one_hot(labels, num_classes).float()  # [B, K]
    W = classifier_weight.detach()                      # [K, D] — detach to avoid graph
    g = torch.matmul(p - y_onehot, W)                   # [B, K] @ [K, D] = [B, D]
    return g


class GradientAlignedMAB:
    """
    Sliding-Window UCB MAB with Gradient Alignment reward.

    Key differences from original OperationMAB:
    1. Reward = cosine similarity (not val loss delta)
    2. Sliding window per epoch (not cumulative)
    3. Z-score normalization for UCB (handles reward scale changes)
    """

    def __init__(self, operations, ucb_c=1.0, window_size=None):
        self.K = len(operations)
        self.operations = operations
        self.op_names = [op.name for op in operations]
        self.ucb_c = ucb_c

        # Current window statistics
        self.counts = np.zeros(self.K)
        self.reward_sums = np.zeros(self.K)

        # Buffer for current epoch (becomes next window)
        self._epoch_counts = np.zeros(self.K)
        self._epoch_reward_sums = np.zeros(self.K)
        self._epoch_rewards_all = []  # for global std computation

    def reset_epoch(self):
        """
        Call at the START of each epoch.
        Slide window: current epoch's stats become the new window.
        """
        self.counts = self._epoch_counts.copy()
        self.reward_sums = self._epoch_reward_sums.copy()
        self._epoch_counts = np.zeros(self.K)
        self._epoch_reward_sums = np.zeros(self.K)
        self._epoch_rewards_all = []

    def select(self):
        """UCB selection with z-score normalization."""
        total = self.counts.sum()

        # Round-robin for first K pulls (ensure every arm tried)
        if total < self.K:
            return int(total) % self.K

        # Compute mean rewards per arm
        means = self.reward_sums / (self.counts + 1e-8)

        # Z-score normalization across arms
        global_mean = means.mean()
        global_std = means.std() + 1e-8
        normalized = (means - global_mean) / global_std

        # UCB exploration term
        exploration = self.ucb_c * np.sqrt(
            np.log(total + 1) / (self.counts + 1e-8)
        )

        ucb_scores = normalized + exploration
        return int(np.argmax(ucb_scores))

    def update(self, arm, reward):
        """Update both current window and epoch buffer."""
        self.counts[arm] += 1
        self.reward_sums[arm] += reward
        self._epoch_counts[arm] += 1
        self._epoch_reward_sums[arm] += reward
        self._epoch_rewards_all.append(reward)

    def stats(self):
        """Pretty print MAB statistics."""
        avg = self.reward_sums / (self.counts + 1e-8)
        total = self.counts.sum()
        info = []
        for i in range(self.K):
            info.append('{}: n={:.0f} cos={:.4f}'.format(
                self.op_names[i], self.counts[i], avg[i]
            ))
        # Also show global reward stats
        if self._epoch_rewards_all:
            all_r = self._epoch_rewards_all
            info.append('epoch_avg_cos={:.4f} std={:.4f}'.format(
                np.mean(all_r), np.std(all_r)
            ))
        return ' | '.join(info)


# ─────────────────────────────────────────────
# EntAugment helpers (unchanged from original)
# ─────────────────────────────────────────────

class EntAugmentFixed:
    def __init__(self, M, op_idx):
        self.op = ALL_TRANSFORMS[op_idx]
        self.level = min(int(PARAMETER_MAX * M) + 1, PARAMETER_MAX)

    def __call__(self, img):
        return self.op.pil_transformer(1., self.level)(img)


def make_magnitude_MAB(magnitude, cutout_length, op_idx):
    trivialaugment.set_augmentation_space(augmentation_space='standard', num_strengths=30)
    return transforms.Compose([
        transforms.ToPILImage(),
        EntAugmentFixed(M=magnitude, op_idx=op_idx),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
        Cutout(1, cutout_length)
    ])


# ─────────────────────────────────────────────
# [AGA-MAB NEW] Training loop
# ─────────────────────────────────────────────

def train(net, epoch, mab, classifier_layer):
    global optimizer
    net.train()

    training_loss = 0.0
    training_magnitude = 0.0
    training_cosine = 0.0
    total = len(train_loader.dataset)
    correct = 0
    epoch_op_counts = np.zeros(mab.K)
    cosine_count = 0

    for i, data in enumerate(train_loader, 0):
        idx, inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # ── Step 1: MAB selects operation ──
        if epoch >= warmup_epoch:
            op_idx = mab.select()
        else:
            op_idx = random.randint(0, mab.K - 1)  # random during warmup

        epoch_op_counts[op_idx] += 1
        trainset.make_magnitude_transform = lambda magnitude, cutout_length, oi=op_idx: \
            make_magnitude_MAB(magnitude, cutout_length, op_idx=oi)

        # ── Step 2: Clean forward pass (analytical gradient, no backward) ──
        # [AGA-MAB NEW] This is the key addition
        if epoch >= warmup_epoch:
            with torch.no_grad():
                outputs_clean = net(inputs)
                p_clean = F.softmax(outputs_clean, dim=1)
                g_clean = compute_analytical_gradient(
                    p_clean, labels, classifier_layer.weight, NUM_CLASSES
                )
                # g_clean shape: [B, D], e.g. [128, 512] for ResNet-18

        # ── Step 3: Augmented forward pass (normal training) ──
        optimizer.zero_grad()
        outputs = net(inputs)  # inputs are augmented by Dataset __getitem__
        loss = criterion(outputs, labels)

        # ── Step 4: Entropy-driven magnitude (same as EntAugment) ──
        probability = F.softmax(outputs, dim=1)
        entropy_val = -torch.sum(probability * torch.log(probability + 1e-8), dim=1)
        entropy_normalized = entropy_val / np.log(NUM_CLASSES)
        magnitude = (1.0 - entropy_normalized).clamp(0.0, 1.0)

        loss_scalar = loss.mean()
        trainset.set_MAGNITUDE(idx, magnitude.detach().cpu())
        training_loss += loss_scalar.item()
        training_magnitude += magnitude.mean().item()

        # ── Step 5: Compute gradient alignment reward ──
        # [AGA-MAB NEW]
        if epoch >= warmup_epoch:
            g_aug = compute_analytical_gradient(
                probability.detach(), labels, classifier_layer.weight, NUM_CLASSES
            )
            # Per-sample cosine similarity, then batch mean
            S_batch = F.cosine_similarity(g_clean, g_aug, dim=1)  # [B]
            S_j = S_batch.mean().item()

            mab.update(op_idx, S_j)
            training_cosine += S_j
            cosine_count += 1

        # ── Step 6: Normal backward + update ──
        _, predicted = outputs.max(1)
        loss_scalar.backward()
        optimizer.step()
        correct += predicted.eq(labels).sum().item()

        # ── Logging ──
        if (i + 1) % args.log_interval == 0:
            trained_total = (i + 1) * len(labels)
            top_op = mab.op_names[int(np.argmax(epoch_op_counts))]
            avg_cos = training_cosine / max(cosine_count, 1)
            print('Train Epoch: {} (top_op={}) [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.4f} Mag: {:.4f} Cos: {:.4f} Acc: {:.2f}'.format(
                epoch, top_op,
                trained_total, total, 100. * trained_total / total,
                training_loss / (i + 1), training_magnitude / (i + 1),
                avg_cos, 100. * correct / trained_total))

    if epoch >= warmup_epoch:
        trainset.is_warmup_finished = True


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


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('=== AGA-MAB: Analytical Gradient-Aligned MAB ===')
    print('Seed: {}, UCB c: {}'.format(args.seed, args.ucb_c))
    print('Dataset: {}, Model: {}, Epochs: {}'.format(
        args.dataset, cfg['model']['type'], epoches))
    print('Num operations: {}'.format(len(ALL_TRANSFORMS)))

    # [AGA-MAB NEW] Get classifier layer for analytical gradient
    classifier_layer = get_classifier_layer(model)
    print('Classifier layer: weight shape = {}'.format(classifier_layer.weight.shape))

    # [AGA-MAB NEW] Create Gradient-Aligned MAB (replaces OperationMAB)
    mab = GradientAlignedMAB(
        ALL_TRANSFORMS,
        ucb_c=args.ucb_c,
        window_size=len(train_loader)
    )

    for epoch in tqdm(range(start_epoch, epoches)):
        # [AGA-MAB NEW] Reset sliding window at epoch start
        if epoch >= warmup_epoch:
            mab.reset_epoch()

        train(model, epoch, mab, classifier_layer)
        test(model, epoch)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print('\n[AGA-MAB Stats @ epoch {}]\n{}\n'.format(epoch, mab.stats()))

    # Auto-save results
    result_file = 'benchmark_results.csv'
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['dataset', 'model', 'seed', 'best_epoch', 'best_acc', 'method'])
        writer.writerow([args.dataset, cfg['model']['type'], args.seed,
                         best_epoch, best_acc, 'AGA_MAB'])
    print('Result saved to {}'.format(result_file))