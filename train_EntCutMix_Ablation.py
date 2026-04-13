import os
import math
import random
import argparse
import csv
import numpy as np

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
parser.add_argument('--config', type=str, required=True,
                    choices=['A', 'H1', 'H3', 'H4', 'H5'],
                    help='A=EntAugment only, H1=EntAug+CutMix(fixed), '
                         'H3=EntAug+CutMix(entropy-sched), '
                         'H4=EntAug+CutMix(linear-sched), '
                         'H5=CutMix alone')
parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                    help='Beta distribution alpha for CutMix lambda')
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
from warmup_scheduler import GradualWarmupScheduler


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)


# ─────────────────────────────────────────────
# Config & Data
# ─────────────────────────────────────────────

with open(args.conf) as f:
    cfg = yaml.safe_load(f)

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
])

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

if args.dataset == 'CIFAR10':
    root = 'data/CIFAR10/'
    trainset = CIFAR10Dataset(root=root, train=True, transform=transform_train, aug=args.aug)
    testset = CIFAR10Dataset(root=root, train=False, transform=transform_test)
elif args.dataset == 'CIFAR100':
    root = 'data/CIFAR100/cifar-100-python/'
    trainset = CIFAR100Dataset(root, train=True, fine_label=True,
                               transform=transform_train, aug=args.aug)
    testset = CIFAR100Dataset(root, train=False, fine_label=True,
                              transform=transform_test)

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
# CutMix helpers
# ─────────────────────────────────────────────

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(inputs, labels, alpha=1.0):
    """Standard CutMix with random lambda from Beta distribution."""
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).cuda()

    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to actual pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))

    return inputs, labels, labels[index], lam


# ─────────────────────────────────────────────
# CutMix probability logic
# ─────────────────────────────────────────────

prev_val_mean_entropy = 1.0  # start at 1.0 = model knows nothing


def get_cutmix_prob(config, epoch, total_epochs):
    global prev_val_mean_entropy

    if config == 'A':
        return 0.0
    elif config in ['H1', 'H5']:
        return 0.5
    elif config == 'H3':
        # Entropy-scheduled: adaptive to model confidence
        # Early training: val_entropy ~ 1.0 → p ~ 0.0
        # Late training:  val_entropy ~ 0.1 → p ~ 0.45
        return 0.5 * (1.0 - prev_val_mean_entropy)
    elif config == 'H4':
        # Linear-scheduled: blind increase
        return 0.5 * (epoch / total_epochs)


def compute_val_entropy(net):
    """Compute mean normalized entropy on test set."""
    global prev_val_mean_entropy
    net.eval()
    entropy_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            outputs = net(inputs)
            probability = F.softmax(outputs, dim=1)
            entropy = -torch.sum(probability * torch.log(probability + 1e-8), dim=1)
            entropy_norm = entropy / np.log(NUM_CLASSES)
            entropy_sum += entropy_norm.sum().item()
            total_samples += labels.size(0)

    prev_val_mean_entropy = entropy_sum / total_samples
    return prev_val_mean_entropy


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(net, epoch):
    global optimizer
    net.train()

    p_cutmix = get_cutmix_prob(args.config, epoch, epoches)

    training_loss = 0.0
    training_magnitude = 0.0
    total = len(train_loader.dataset)
    correct = 0
    cutmix_count = 0

    for i, data in enumerate(train_loader, 0):
        idx, inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        use_cutmix = np.random.random() < p_cutmix

        if use_cutmix:
            cutmix_count += 1
            inputs, labels_a, labels_b, lam = cutmix_data(
                inputs, labels, alpha=args.cutmix_alpha
            )
            outputs = net(inputs)
            loss = lam * criterion(outputs, labels_a).mean() \
                 + (1 - lam) * criterion(outputs, labels_b).mean()
            # CutMix batch: do NOT update magnitude (entropy is noisy)

        else:
            outputs = net(inputs)
            loss = criterion(outputs, labels).mean()

            # H5 = CutMix alone, use random magnitude
            if args.config == 'H5':
                magnitude = torch.rand(len(idx))
                trainset.set_MAGNITUDE(idx, magnitude)
            else:
                probability = F.softmax(outputs, dim=1)
                entropy_val = -torch.sum(
                    probability * torch.log(probability + 1e-8), dim=1
                )
                entropy_norm = entropy_val / np.log(NUM_CLASSES)
                magnitude = (1.0 - entropy_norm).clamp(0.0, 1.0)
                trainset.set_MAGNITUDE(idx, magnitude.detach().cpu())
                training_magnitude += magnitude.mean().item()

        training_loss += loss.item()

        _, predicted = outputs.max(1)
        loss.backward()
        optimizer.step()

        if use_cutmix:
            correct += (lam * predicted.eq(labels_a).sum().float()
                      + (1 - lam) * predicted.eq(labels_b).sum().float()).item()
        else:
            correct += predicted.eq(labels).sum().item()

        if (i + 1) % args.log_interval == 0:
            trained_total = (i + 1) * len(inputs)
            n_clean = (i + 1) - cutmix_count
            mag_mean = training_magnitude / max(n_clean, 1)
            print('Train Epoch: {} [Config={}] [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.4f} Mag: {:.4f} p_cutmix: {:.3f} Acc: {:.2f}'.format(
                epoch, args.config,
                trained_total, total, 100. * trained_total / total,
                training_loss / (i + 1), mag_mean,
                p_cutmix, 100. * correct / trained_total))

    if epoch >= warmup_epoch:
        trainset.is_warmup_finished = True


# ─────────────────────────────────────────────
# Test loop
# ─────────────────────────────────────────────

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
    config_desc = {
        'A':  'EntAugment only (p_cutmix=0)',
        'H1': 'EntAugment + CutMix fixed (p=0.5)',
        'H3': 'EntAugment + CutMix entropy-scheduled (p=0.5*(1-val_ent))',
        'H4': 'EntAugment + CutMix linear-scheduled (p=0.5*epoch/total)',
        'H5': 'CutMix alone fixed (p=0.5, no EntAugment)',
    }
    print('=== EntCutMix Ablation ===')
    print('Config: {} — {}'.format(args.config, config_desc[args.config]))
    print('Seed: {}, Dataset: {}, Model: {}, Epochs: {}'.format(
        args.seed, args.dataset, cfg['model']['type'], epoches))

    for epoch in tqdm(range(start_epoch, epoches)):
        train(model, epoch)
        test(model, epoch)

        # Update validation entropy for H3
        if args.config == 'H3':
            val_ent = compute_val_entropy(model)
            print('[Val Entropy: {:.4f}, Next p_cutmix: {:.4f}]'.format(
                val_ent, 0.5 * (1.0 - val_ent)))

        scheduler.step()

    # Save results
    result_file = 'ablation_results.csv'
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['config', 'dataset', 'model', 'seed',
                             'best_epoch', 'best_acc'])
        writer.writerow([args.config, args.dataset, cfg['model']['type'],
                         args.seed, best_epoch, best_acc])
    print('Result saved to {}'.format(result_file))