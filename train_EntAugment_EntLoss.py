import pathlib
import sys
import os
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument('--log_interval',type=int,default=50,help='log training status')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--conf', default='./confs/resnet18.yaml', type=str,  help=' yaml file')
parser.add_argument('--gpus',type=str,default='0')
parser.add_argument('--resume',type=str,default=None)
parser.add_argument('--cutout_length',type=int, default=16)
parser.add_argument('--dataset',type=str, required=True)
parser.add_argument('--save_model',type=bool, default=False)
parser.add_argument('--num_worker',type=int, default=8,choices=[2,4,8,16,32])
parser.add_argument('--aug',type=str, default='entaugment')
parser.add_argument('--alpha_schedule', type=str, default='linear',
                    choices=['linear', 'cosine', 'step'])
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

from typing import Generator
import numpy as np
import math
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pickle
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
import random
from Dataset import CIFAR10Dataset, CIFAR100Dataset
from Network import *
from organize_transform import make_transform
import yaml
from warmup_scheduler import GradualWarmupScheduler
import pandas as pd
import copy
from scipy.stats import entropy as scipy_entropy
import threading

cuda = True if torch.cuda.is_available() else False

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
momentum = args.momentum
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
        momentum=momentum,
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
batch = cfg['batch']
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


def dual_signal_magnitude(entropy_normalized, margin, alpha):
    # Pure entropy-only magnitude (same as original EntAugment)
    # Ignore margin and alpha completely
    mag = 1.0 - entropy_normalized
    return mag.clamp(0.0, 1.0)


def train(net, epoch):
    global scheduler, optimizer
    global loss
    net.train()
    training_loss = 0.0
    training_magnitude = 0.0
    total = len(train_loader.dataset)
    correct = 0

    for i, data in enumerate(train_loader, 0):
        idx, inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        probability = F.softmax(outputs, dim=1)
        entropy_val = -torch.sum(probability * torch.log(probability + 1e-8), dim=1)
        entropy_normalized = entropy_val / np.log(NUM_CLASSES)

        top2_probs, _ = torch.topk(probability, k=2, dim=1)
        margin = top2_probs[:, 0] - top2_probs[:, 1]

        magnitude = dual_signal_magnitude(
            entropy_normalized.detach(), margin.detach(), alpha=None
        )

        # EntLoss: minimize entropy → model more confident
        ce_loss = loss.mean()
        ent_loss = torch.sum(
            probability * torch.log(probability + 1e-8), dim=1
        ).mean() / math.log(NUM_CLASSES)
        loss = ce_loss + ent_loss

        trainset.set_MAGNITUDE(idx, magnitude.detach().cpu())
        training_loss += loss.item()
        training_magnitude += magnitude.mean().item()

        _, predicted = outputs.max(1)
        loss.backward()
        optimizer.step()
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % args.log_interval == 0:
            trained_total = (i + 1) * len(labels)
            acc = 100. * correct / trained_total
            progress = 100. * trained_total / total
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.4f} Mag: {:.4f} Acc: {:.2f}'.format(
                epoch, trained_total, total, progress,
                training_loss / (i + 1), training_magnitude / (i + 1), acc))

    if epoch >= warmup_epoch:
        trainset.is_warmup_finished = True


def test(net, epoch):
    global best_acc, best_epoch
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
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
    print('=== Config E: EntAugment + EntLoss (entropy-only magnitude) ===')
    print('Dataset: {}, Model: {}, Epochs: {}'.format(
        args.dataset, cfg['model']['type'], epoches))
    for epoch in tqdm(range(start_epoch, epoches)):
        train(model, epoch)
        test(model, epoch)
        scheduler.step()