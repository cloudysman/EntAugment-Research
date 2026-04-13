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
parser.add_argument('--gpus', type=str, default='6,7')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--cutout_length', type=int, default=16)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--num_worker', type=int, default=8, choices=[2, 4, 8, 16, 32])
parser.add_argument('--aug', type=str, default='entaugment')
parser.add_argument('--alpha_schedule', type=str, default='linear',
                    choices=['linear', 'cosine', 'step'],
                    help='Schedule for alpha decay in dual-signal magnitude')
parser.add_argument('--ucb_alpha', type=float, default=1.0,
                    help='UCB exploration constant for MAB (higher = more exploration)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

import numpy as np
import math
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

cuda = True if torch.cuda.is_available() else False

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
# MAB: Multi-Armed Bandit with UCB selection
# ─────────────────────────────────────────────

class OperationMAB:
    """
    UCB1 bandit over augmentation operations.
    Each arm = one augmentation operation from ALL_TRANSFORMS.
    Reward = reduction in validation loss after an epoch with that operation.
    """
    def __init__(self, operations, ucb_alpha=1.0):
        self.K = len(operations)
        self.operations = operations
        self.op_names = [op.name for op in operations]
        self.counts = np.zeros(self.K)    # times each arm was selected
        self.rewards = np.zeros(self.K)   # cumulative reward per arm
        self.ucb_alpha = ucb_alpha
        self.current_op = 0               # op selected for current epoch

    def select(self, epoch):
        """
        UCB1 selection.
        For the first K epochs, try each operation once (initialization).
        After that, use UCB scores.
        """
        if epoch < self.K:
            # Round-robin init: ensure every arm is tried at least once
            return epoch % self.K

        total = self.counts.sum()
        avg_reward = self.rewards / (self.counts + 1e-8)
        exploration = self.ucb_alpha * np.sqrt(np.log(total + 1) / (self.counts + 1e-8))
        ucb_scores = avg_reward + exploration
        return int(np.argmax(ucb_scores))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward

    def stats(self):
        avg = self.rewards / (self.counts + 1e-8)
        info = []
        for i in range(self.K):
            info.append('{}: cnt={:.0f} avg_r={:.4f}'.format(
                self.op_names[i], self.counts[i], avg[i]))
        return ' | '.join(info)


# ─────────────────────────────────────────────
# EntAugment with fixed operation (for MAB)
# ─────────────────────────────────────────────

class EntAugmentFixed:
    """
    Same as EntAugment but uses the MAB-selected operation
    instead of random.choices.
    """
    def __init__(self, M, op_idx):
        self.op = ALL_TRANSFORMS[op_idx]
        self.level = min(int(PARAMETER_MAX * M) + 1, PARAMETER_MAX)

    def __call__(self, img):
        return self.op.pil_transformer(1., self.level)(img)


def make_magnitude_MAB(magnitude, cutout_length, op_idx):
    """
    Build per-sample transform using MAB-selected operation at given magnitude.
    """
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
# Dual-Signal Magnitude (Phase 1)
# ─────────────────────────────────────────────

def get_alpha(epoch, total_epochs, schedule='linear'):
    progress = epoch / total_epochs
    if schedule == 'linear':
        return 1.0 - progress
    elif schedule == 'cosine':
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == 'step':
        if progress < 0.33:
            return 1.0
        elif progress < 0.66:
            return 0.5
        else:
            return 0.0


def dual_signal_magnitude(entropy_normalized, margin, alpha):
    h_entropy = 1.0 - entropy_normalized          # confident → augment strongly
    h_margin = margin                              # clear top-2 gap → augment strongly
    mag = alpha * h_entropy + (1.0 - alpha) * h_margin
    return mag.clamp(0.0, 1.0)


# ─────────────────────────────────────────────
# Val loss for MAB reward
# ─────────────────────────────────────────────

def compute_val_loss(net):
    """
    Compute loss on one mini-batch from test_loader.
    Used as a fast proxy for MAB reward (reward = prev_loss - current_loss).
    """
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

def train(net, epoch, op_idx):
    global optimizer
    net.train()

    # Patch dataset to use MAB-selected operation for this epoch
    trainset.make_magnitude_transform = lambda magnitude, cutout_length: \
        make_magnitude_MAB(magnitude, cutout_length, op_idx=op_idx)

    alpha = get_alpha(epoch, epoches, schedule=args.alpha_schedule)
    training_loss = 0.0
    training_magnitude = 0.0
    training_entropy = 0.0
    training_margin = 0.0
    total = len(train_loader.dataset)
    correct = 0

    for i, data in enumerate(train_loader, 0):
        idx, inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Entropy signal
        probability = F.softmax(outputs, dim=1)
        entropy_val = -torch.sum(probability * torch.log(probability + 1e-8), dim=1)
        entropy_normalized = entropy_val / np.log(NUM_CLASSES)

        # Margin signal
        top2_probs, _ = torch.topk(probability, k=2, dim=1)
        margin = top2_probs[:, 0] - top2_probs[:, 1]

        # Dual-signal magnitude
        magnitude = dual_signal_magnitude(
            entropy_normalized.detach(), margin.detach(), alpha
        )

        loss = loss.mean()
        trainset.set_MAGNITUDE(idx, magnitude.detach().cpu())
        training_loss += loss.item()
        training_magnitude += magnitude.mean().item()
        training_entropy += entropy_normalized.mean().item()
        training_margin += margin.mean().item()

        _, predicted = outputs.max(1)
        loss.backward()
        optimizer.step()
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % args.log_interval == 0:
            trained_total = (i + 1) * len(labels)
            print('Train Epoch: {} (op={}) [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.4f} Mag: {:.4f} Entropy: {:.4f} Margin: {:.4f} Alpha: {:.3f} Acc: {:.2f}'.format(
                epoch, ALL_TRANSFORMS[op_idx].name,
                trained_total, total, 100. * trained_total / total,
                training_loss / (i + 1), training_magnitude / (i + 1),
                training_entropy / (i + 1), training_margin / (i + 1),
                alpha, 100. * correct / trained_total))

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
    print('=== MAB + Dual-Signal Magnitude Training ===')
    print('Num operations: {}'.format(len(ALL_TRANSFORMS)))
    print('Operations: {}'.format([op.name for op in ALL_TRANSFORMS]))
    print('UCB alpha: {}, Alpha schedule: {}'.format(args.ucb_alpha, args.alpha_schedule))
    print('Dataset: {}, Model: {}, Epochs: {}'.format(
        args.dataset, cfg['model']['type'], epoches))

    mab = OperationMAB(ALL_TRANSFORMS, ucb_alpha=args.ucb_alpha)
    prev_val_loss = compute_val_loss(model)

    for epoch in tqdm(range(start_epoch, epoches)):
        # MAB selects operation for this epoch
        op_idx = mab.select(epoch)
        mab.current_op = op_idx

        train(model, epoch, op_idx)

        # Compute reward = reduction in val loss
        current_val_loss = compute_val_loss(model)
        reward = prev_val_loss - current_val_loss
        mab.update(op_idx, reward)
        prev_val_loss = current_val_loss

        test(model, epoch)
        scheduler.step()

        # Print MAB stats every 10 epochs
        if (epoch + 1) % 10 == 0:
            print('\n[MAB Stats @ epoch {}]\n{}\n'.format(epoch, mab.stats()))
