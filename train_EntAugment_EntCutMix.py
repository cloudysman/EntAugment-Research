import pathlib
import sys
import os
import argparse
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
parser.add_argument('--num_worker', type=int, default=8, choices=[2, 4, 8, 16, 32])
parser.add_argument('--aug', type=str, default='entaugment')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cutmix_prob', type=float, default=0.5,
                    help='Probability of applying CutMix per batch')
parser.add_argument('--result_file', type=str, default='benchmark_composition_results.csv')
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
from organize_transform import make_transform
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

transform = transforms.Compose([
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
    trainset = CIFAR10Dataset(root=root, train=True, transform=transform, aug=args.aug)
    testset = CIFAR10Dataset(root=root, train=False, transform=transform_test)
elif args.dataset == 'CIFAR100':
    root = 'data/CIFAR100/cifar-100-python/'
    trainset = CIFAR100Dataset(root, train=True, fine_label=True,
                               transform=transform, aug=args.aug)
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


def entropy_cutmix_data(inputs, labels, net, num_classes):
    """
    Entropy-Guided CutMix:
    - Cả hai dễ (entropy thấp) → mix mạnh (lambda gần 0.5)
    - Một sample khó → lambda lệch về phía sample dễ (giữ sample khó)
    - Cả hai khó → gần như không mix (lambda gần 1.0)

    Curriculum learning tự nhiên:
    - Đầu training: entropy cao → mix nhẹ
    - Cuối training: entropy thấp → mix mạnh
    """
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).cuda()

    # Compute entropy for each sample (no gradient needed)
    with torch.no_grad():
        outputs = net(inputs)
        probability = F.softmax(outputs, dim=1)
        entropy = -torch.sum(probability * torch.log(probability + 1e-8), dim=1)
        entropy_norm = entropy / np.log(num_classes)  # [0, 1]

    # Confidence = 1 - entropy
    conf_A = (1.0 - entropy_norm)           # [B]
    conf_B = (1.0 - entropy_norm[index])    # [B], shuffled

    # Mix intensity: cả hai phải dễ mới mix mạnh
    mix_intensity = conf_A * conf_B  # [0, 1]

    # Lambda: sample nào dễ hơn bị cắt nhiều hơn
    # conf_A / (conf_A + conf_B) → lớn khi A dễ hơn B → A bị cắt nhiều → B được giữ
    lambda_ratio = conf_A / (conf_A + conf_B + 1e-8)

    # Final lambda: scale by mix_intensity
    # mix_intensity thấp → lambda gần 1 (giữ nguyên A, không mix)
    lam_per_sample = mix_intensity * lambda_ratio + (1 - mix_intensity) * 1.0
    lam = lam_per_sample.mean().item()  # dùng 1 lambda cho toàn batch (CutMix cần 1 bbox)

    # Apply CutMix with computed lambda
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to actual pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))

    labels_a = labels
    labels_b = labels[index]
    return inputs, labels_a, labels_b, lam


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(net, epoch):
    global scheduler, optimizer
    net.train()
    training_loss = 0.0
    training_magnitude = 0.0
    training_lam = 0.0
    cutmix_count = 0
    total = len(train_loader.dataset)
    correct = 0

    for i, data in enumerate(train_loader, 0):
        idx, inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        use_cutmix = np.random.random() < args.cutmix_prob

        if use_cutmix:
            inputs, labels_a, labels_b, lam = entropy_cutmix_data(
                inputs, labels, net, NUM_CLASSES
            )
            outputs = net(inputs)
            loss = lam * criterion(outputs, labels_a).mean() \
                 + (1 - lam) * criterion(outputs, labels_b).mean()
            training_lam += lam
            cutmix_count += 1
        else:
            outputs = net(inputs)
            loss = criterion(outputs, labels).mean()

        # EntAugment magnitude
        probability = F.softmax(outputs, dim=1)
        entropy_val = -torch.sum(probability * torch.log(probability + 1e-8), dim=1)
        entropy_normalized = entropy_val / np.log(NUM_CLASSES)
        magnitude = (1.0 - entropy_normalized).clamp(0.0, 1.0)

        trainset.set_MAGNITUDE(idx, magnitude.detach().cpu())
        training_loss += loss.item()
        training_magnitude += magnitude.mean().item()

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
            avg_lam = training_lam / max(cutmix_count, 1)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.4f} Mag: {:.4f} AvgLam: {:.3f} Acc: {:.2f}'.format(
                epoch, trained_total, total, 100. * trained_total / total,
                training_loss / (i + 1), training_magnitude / (i + 1),
                avg_lam, 100. * correct / trained_total))

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


if __name__ == '__main__':
    print('=== H2: EntAugment + Entropy-Guided CutMix ===')
    print('Seed: {}, cutmix_prob: {}'.format(args.seed, args.cutmix_prob))
    print('Dataset: {}, Model: {}, Epochs: {}'.format(
        args.dataset, cfg['model']['type'], epoches))

    for epoch in tqdm(range(start_epoch, epoches)):
        train(model, epoch)
        test(model, epoch)
        scheduler.step()

    result_file = args.result_file
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['dataset', 'model', 'seed', 'best_epoch', 'best_acc', 'method'])
        writer.writerow([args.dataset, cfg['model']['type'], args.seed,
                         best_epoch, best_acc, 'EntAugment_EntCutMix'])
    print('Result saved to {}'.format(result_file))