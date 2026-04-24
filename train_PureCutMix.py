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
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--num_worker', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cutmix_alpha', type=float, default=1.0)
parser.add_argument('--result_file', type=str, default='benchmark_composition_results.csv')
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

# Pure transform: chỉ crop + flip, KHÔNG EntAugment
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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
    trainset = CIFAR10Dataset(root=root, train=True, transform=transform_train, aug='none')
    testset = CIFAR10Dataset(root=root, train=False, transform=transform_test)
elif args.dataset == 'CIFAR100':
    root = 'data/CIFAR100/cifar-100-python/'
    trainset = CIFAR100Dataset(root, train=True, fine_label=True,
                               transform=transform_train, aug='none')
    testset = CIFAR100Dataset(root, train=False, fine_label=True,
                              transform=transform_test)

# Bypass EntAugment hoàn toàn: dùng external_transform (crop+flip chỉ)
trainset.external_transform = transform_train

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
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
    return inputs, labels, labels[index], lam


# ─────────────────────────────────────────────
# Training loop — Pure CutMix, no EntAugment
# ─────────────────────────────────────────────

def train(net, epoch):
    global optimizer
    net.train()
    training_loss = 0.0
    total = len(train_loader.dataset)
    correct = 0

    for i, data in enumerate(train_loader, 0):
        idx, inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        use_cutmix = np.random.random() < 0.5

        if use_cutmix:
            inputs, labels_a, labels_b, lam = cutmix_data(
                inputs, labels, alpha=args.cutmix_alpha
            )
            outputs = net(inputs)
            loss = lam * criterion(outputs, labels_a).mean() \
                 + (1 - lam) * criterion(outputs, labels_b).mean()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, labels).mean()

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
            print('Train Epoch: {} [PureCutMix] [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.4f} Acc: {:.2f}'.format(
                epoch, trained_total, total, 100. * trained_total / total,
                training_loss / (i + 1), 100. * correct / trained_total))


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
    print('=== Pure CutMix Baseline (no EntAugment) ===')
    print('Seed: {}, Dataset: {}, Model: {}, Epochs: {}'.format(
        args.seed, args.dataset, cfg['model']['type'], epoches))

    for epoch in tqdm(range(start_epoch, epoches)):
        train(model, epoch)
        test(model, epoch)
        scheduler.step()

    result_file = args.result_file
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['config', 'dataset', 'model', 'seed', 'best_epoch', 'best_acc'])
        writer.writerow(['PureCutMix', args.dataset, cfg['model']['type'],
                         args.seed, best_epoch, best_acc])
    print('Result saved to {}'.format(result_file))