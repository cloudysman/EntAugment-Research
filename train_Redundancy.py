import os
import math
import random
import argparse
import csv
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--conf', default='./confs/resnet18.yaml', type=str)
parser.add_argument('--gpus', type=str, default='6,7')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--cutout_length', type=int, default=16)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--num_worker', type=int, default=8, choices=[2, 4, 8, 16, 32])
parser.add_argument('--aug', type=str, default='entaugment')
# Phase 1 (dual-signal)
parser.add_argument('--alpha_schedule', type=str, default='linear',
                    choices=['linear', 'cosine', 'step'])
# Phase 2 (MAB)
parser.add_argument('--ucb_alpha', type=float, default=1.0,
                    help='UCB exploration constant')
# Phase 3 (redundancy)
parser.add_argument('--redundancy_update_freq', type=int, default=20,
                    help='Recompute redundancy scores every N epochs')
parser.add_argument('--redundancy_threshold', type=float, default=0.95,
                    help='Cosine similarity threshold to count a neighbor')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch as _torch
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False

set_seed(args.seed)

from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from Dataset import CIFAR10Dataset, CIFAR100Dataset
from Network import *
from augmentation.entaugment import ALL_TRANSFORMS, PARAMETER_MAX
from augmentation.cutout import Cutout
from augmentation import trivialaugment
import yaml
from warmup_scheduler import GradualWarmupScheduler

with open(args.conf) as f:
    cfg = yaml.safe_load(f)

NORMALIZE = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
)
transform_test = transforms.Compose([transforms.ToTensor(), NORMALIZE])
transform_warmup = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(), NORMALIZE
])
# Clean transform for feature extraction (no augmentation)
transform_clean = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), NORMALIZE
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
        model.parameters(), lr=cfg['lr'], momentum=args.momentum,
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
    testset  = CIFAR10Dataset(root=root, train=False, transform=transform_test)
elif args.dataset == 'CIFAR100':
    root = 'data/CIFAR100/cifar-100-python/'
    trainset = CIFAR100Dataset(root, train=True, fine_label=True,
                               transform=transform_warmup, aug=args.aug)
    testset  = CIFAR100Dataset(root, train=False, fine_label=True,
                               transform=transform_test)

train_loader = DataLoader(trainset, batch_size=cfg['batch'],
                          shuffle=True, num_workers=8, pin_memory=True)
test_loader  = DataLoader(testset,  batch_size=cfg['batch'],
                          shuffle=False, num_workers=8, pin_memory=True)
# Dedicated loader for feature extraction: shuffle=False, same indices every call
feat_loader  = DataLoader(trainset, batch_size=256,
                          shuffle=False, num_workers=4, pin_memory=True)

start_epoch = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}/{}'.format(args.dataset, args.resume))
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# ─────────────────────────────────────────────
# Phase 2: MAB
# ─────────────────────────────────────────────

class OperationMAB:
    def __init__(self, operations, ucb_alpha=1.0):
        self.K = len(operations)
        self.operations = operations
        self.op_names = [op.name for op in operations]
        self.counts = np.zeros(self.K)
        self.rewards = np.zeros(self.K)
        self.ucb_alpha = ucb_alpha
        self.current_op = 0

    def select(self, epoch):
        if epoch < self.K:
            return epoch % self.K
        total = self.counts.sum()
        avg_reward = self.rewards / (self.counts + 1e-8)
        exploration = self.ucb_alpha * np.sqrt(np.log(total + 1) / (self.counts + 1e-8))
        return int(np.argmax(avg_reward + exploration))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward

    def stats(self):
        avg = self.rewards / (self.counts + 1e-8)
        return ' | '.join('{}: n={:.0f} r={:.4f}'.format(
            self.op_names[i], self.counts[i], avg[i]) for i in range(self.K))


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
        transforms.ToTensor(), NORMALIZE,
        Cutout(1, cutout_length)
    ])


# ─────────────────────────────────────────────
# Phase 1: Dual-signal magnitude
# ─────────────────────────────────────────────

def get_alpha(epoch, total_epochs, schedule='linear'):
    progress = epoch / total_epochs
    if schedule == 'linear':
        return 1.0 - progress
    elif schedule == 'cosine':
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == 'step':
        return 1.0 if progress < 0.33 else (0.5 if progress < 0.66 else 0.0)


def dual_signal_magnitude(entropy_normalized, margin, alpha):
    h_entropy = 1.0 - entropy_normalized
    h_margin  = margin
    return (alpha * h_entropy + (1.0 - alpha) * h_margin).clamp(0.0, 1.0)


# ─────────────────────────────────────────────
# Phase 3: Redundancy-aware scheduling
# ─────────────────────────────────────────────

def extract_features(net):
    """
    Extract feature vectors for all training samples via a hook on m.linear.
    Works for any architecture (ResNet, WideResNet, etc.).
    Temporarily uses clean transform (no augmentation) for consistency.
    Returns: features [N, D], labels [N], indices [N]  (all on CPU)
    """
    m = net.module if hasattr(net, 'module') else net

    # Patch trainset to return clean (non-augmented) images
    saved_make_mag  = trainset.make_magnitude_transform
    saved_is_warmup = trainset.is_warmup_finished
    trainset.is_warmup_finished = True
    trainset.make_magnitude_transform = lambda magnitude, cutout_length: transform_clean

    all_feats   = []
    all_labels  = []
    all_indices = []

    # Hook captures input to the final linear layer — architecture-agnostic
    def _hook(module, inp, out):
        all_feats.append(inp[0].cpu())

    hook = m.linear.register_forward_hook(_hook)

    net.eval()
    with torch.no_grad():
        for batch in feat_loader:
            idx, inputs, labels = batch
            inputs = inputs.cuda()
            net(inputs)
            all_labels.append(labels)
            all_indices.append(idx)

    hook.remove()

    # Restore trainset state
    trainset.make_magnitude_transform = saved_make_mag
    trainset.is_warmup_finished = saved_is_warmup

    return (torch.cat(all_feats, dim=0),
            torch.cat(all_labels, dim=0),
            torch.cat(all_indices, dim=0))


def compute_redundancy(net, threshold=0.95):
    """
    For each training sample, count how many same-class neighbors
    have cosine similarity > threshold. Normalize to [0,1].

    Returns: redundancy_scores [N_train] tensor indexed by sample index.
    """
    print('[Redundancy] Extracting features...')
    feats, labels, indices = extract_features(net)

    # L2-normalize for cosine similarity via dot product
    feats = F.normalize(feats, dim=1)

    n_train = len(trainset)
    raw_counts = torch.zeros(n_train)

    print('[Redundancy] Computing per-class similarity matrices...')
    for cls in range(NUM_CLASSES):
        cls_mask = (labels == cls)
        cls_feats   = feats[cls_mask].cuda()     # [N_cls, 512]
        cls_indices = indices[cls_mask]

        # Cosine similarity matrix: [N_cls, N_cls]
        sim_matrix = cls_feats @ cls_feats.T
        # Count neighbors above threshold (exclude self by subtracting 1)
        counts = (sim_matrix > threshold).sum(dim=1).float() - 1
        counts = counts.clamp(min=0).cpu()

        for i, sample_idx in enumerate(cls_indices):
            raw_counts[sample_idx.item()] = counts[i].item()

    # Normalize to [0, 1] using global max
    max_count = raw_counts.max().item()
    if max_count > 0:
        redundancy_scores = raw_counts / max_count
    else:
        redundancy_scores = raw_counts

    n_redundant = (redundancy_scores > 0.5).sum().item()
    print('[Redundancy] Done. Max neighbors: {:.0f}, '
          'Samples with score>0.5: {}/{}'.format(
        max_count, n_redundant, n_train))

    return redundancy_scores  # [N_train], indexed by sample index


def get_redundancy_weight(epoch, total_epochs):
    """
    Redundancy weight grows from 0 → 1 over training.
    Early: don't care (features not converged yet).
    Late:  reduce augmentation on redundant samples.
    """
    return epoch / total_epochs


# ─────────────────────────────────────────────
# Val loss helper for MAB reward
# ─────────────────────────────────────────────

def compute_val_loss(net):
    net.eval()
    val_criterion = nn.CrossEntropyLoss()
    inputs, targets = next(iter(test_loader))
    with torch.no_grad():
        outputs = net(inputs.cuda())
        loss = val_criterion(outputs, targets.cuda())
    return loss.item()


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(net, epoch, op_idx, redundancy_scores):
    global optimizer
    net.train()

    # Patch dataset: MAB-selected operation
    trainset.make_magnitude_transform = lambda magnitude, cutout_length: \
        make_magnitude_MAB(magnitude, cutout_length, op_idx=op_idx)

    alpha             = get_alpha(epoch, epoches, schedule=args.alpha_schedule)
    redundancy_weight = get_redundancy_weight(epoch, epoches)

    training_loss      = 0.0
    training_magnitude = 0.0
    total              = len(train_loader.dataset)
    correct            = 0

    for i, data in enumerate(train_loader, 0):
        idx, inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Phase 1: dual-signal
        probability       = F.softmax(outputs, dim=1)
        entropy_val       = -torch.sum(probability * torch.log(probability + 1e-8), dim=1)
        entropy_normalized = entropy_val / np.log(NUM_CLASSES)
        top2_probs, _     = torch.topk(probability, k=2, dim=1)
        margin            = top2_probs[:, 0] - top2_probs[:, 1]
        magnitude         = dual_signal_magnitude(
            entropy_normalized.detach(), margin.detach(), alpha
        )

        # Phase 3: adjust magnitude by redundancy
        if redundancy_scores is not None:
            r = redundancy_scores[idx].cuda()             # [B], range [0,1]
            magnitude = magnitude * (1.0 - redundancy_weight * r)
            magnitude = magnitude.clamp(0.0, 1.0)

        ce_loss = loss.mean()
        ent_loss = torch.sum(probability * torch.log(probability + 1e-8), dim=1).mean()
        ent_loss = ent_loss / math.log(NUM_CLASSES)
        loss = ce_loss + ent_loss
        trainset.set_MAGNITUDE(idx, magnitude.detach().cpu())
        training_loss      += loss.item()
        training_magnitude += magnitude.mean().item()

        _, predicted = outputs.max(1)
        loss.backward()
        optimizer.step()
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % args.log_interval == 0:
            trained_total = (i + 1) * len(labels)
            print('Train Epoch: {} (op={}) [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.4f} Mag: {:.4f} Alpha: {:.3f} RedWeight: {:.3f} Acc: {:.2f}'.format(
                epoch, ALL_TRANSFORMS[op_idx].name,
                trained_total, total, 100. * trained_total / total,
                training_loss / (i + 1), training_magnitude / (i + 1),
                alpha, redundancy_weight, 100. * correct / trained_total))

    if epoch >= warmup_epoch:
        trainset.is_warmup_finished = True


def test(net, epoch):
    global best_acc, best_epoch
    net.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct * 100. / total
    print('EPOCH:{}, ======================ACC:{}===================='.format(epoch, acc))
    acc_list.append(acc)
    if acc >= best_acc:
        best_acc   = acc
        best_epoch = epoch
    print('BEST EPOCH:{},BEST ACC:{}'.format(best_epoch, best_acc))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Phase 1+2+3: Dual-Signal + MAB + Redundancy-Aware ===')
    print('Redundancy: update_freq={}, threshold={}'.format(
        args.redundancy_update_freq, args.redundancy_threshold))
    print('UCB alpha: {}, Alpha schedule: {}'.format(args.ucb_alpha, args.alpha_schedule))
    print('Dataset: {}, Model: {}, Epochs: {}'.format(
        args.dataset, cfg['model']['type'], epoches))

    mab = OperationMAB(ALL_TRANSFORMS, ucb_alpha=args.ucb_alpha)
    prev_val_loss     = compute_val_loss(model)
    redundancy_scores = None  # computed lazily after warmup

    for epoch in tqdm(range(start_epoch, epoches)):

        # Phase 3: recompute redundancy every N epochs (skip during warmup)
        if epoch >= warmup_epoch and epoch % args.redundancy_update_freq == 0:
            redundancy_scores = compute_redundancy(
                model, threshold=args.redundancy_threshold
            )

        # Phase 2: MAB selects operation
        op_idx = mab.select(epoch)
        mab.current_op = op_idx

        train(model, epoch, op_idx, redundancy_scores)

        # MAB reward = reduction in val loss
        current_val_loss = compute_val_loss(model)
        mab.update(op_idx, prev_val_loss - current_val_loss)
        prev_val_loss = current_val_loss

        test(model, epoch)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print('\n[MAB Stats @ epoch {}]\n{}\n'.format(epoch, mab.stats()))

    result_file = 'results.csv'
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['dataset', 'model', 'seed', 'best_epoch', 'best_acc', 'method'])
        writer.writerow([args.dataset, cfg['model']['type'], args.seed, best_epoch, best_acc, 'proposed'])
