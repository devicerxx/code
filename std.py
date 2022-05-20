import argparse
import os
import time
import numpy as np
import random
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from utils import Logger, set_seed, get_model
from advertorch.utils import NormalizeByChannelMeanStd
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import wandb
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

########################## parse arguments ##########################
parser = argparse.ArgumentParser(description='Std training for flooding')
parser.add_argument('--EXP', metavar='EXP', default='Flooding_exp_stand_', help='experiment name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='PreActResNet18',
                    help='model architecture (default: PreActResNet18)')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='training datasets')
parser.add_argument('--optimizer', metavar='OPTIMIZER', default='sgd', type=str,
                    help='optimizer for training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50 iterations)')
parser.add_argument('--log-dir', dest='log_dir',
                    help='The directory used to save the log',
                    default='save_temp', type=str)
parser.add_argument('--log-name', dest='log_name',
                    help='The log file name',
                    default='log', type=str)
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--randomseed',
                    help='Randomseed for training and initialization',
                    type=int, default=0)


# Record training statistics
arr_time = []

def main():
    global args
    global arr_time

    args = parser.parse_args()

    # Check the save_dir exists or not
    print('save dir:', args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Check the log_dir exists or not
    print('log dir:', args.log_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    sys.stdout = Logger(os.path.join(args.log_dir, args.log_name))
    print('tracking with wandb!')

    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'test acc',
            'goal': 'maximize'
        }
    }

    parameters_dict = {
        'b': {
            'values': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="subAT", entity="starry_sky")

    wandb.agent(sweep_id, model_pipeline, count=10)

def print_args(config):
    print('batch size:', config.batch_size)
    print('Model:', config.arch)
    print('Dataset:', config.datasets)


def get_datasets(config):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = CIFAR10('../../Downloads/dataset/cifar10', train=True, transform=train_transform, download=True)
    test_set = CIFAR10('../../Downloads/dataset/cifar10', train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def model_pipeline(hyperparameters=None):
    # tell wandb to get started
    global args

    with wandb.init(config=hyperparameters):
        wandb.config.update(args)
        config = wandb.config

        # name a exp
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        wandb.run.name = args.EXP + str(config.b) + '_' + date

        print_args(config)
        print('random seed:', config.randomseed)
        set_seed(config.randomseed)

        # Define model
        model = torch.nn.DataParallel(get_model(config))
        model.cuda()

        cudnn.benchmark = True

        # Prepare Dataloader
        train_loader, test_loader = get_datasets(config)

        # Define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()

        if config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=config.momentum,
                                        weight_decay=config.weight_decay)

        print('Start training: ', config.start_epoch, '->', config.epochs)

        for epoch in range(config.start_epoch, config.epochs):
            # train for one epoch
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            train(train_loader, model, criterion, optimizer, epoch, config)

            # evaluate on test set
            natural_acc, natural_loss = validate(test_loader, model, criterion, config)

            wandb.log({"test acc": natural_acc, "test loss": natural_loss, 'epoch': epoch})


def train(train_loader, model, criterion, optimizer, epoch, config):
    """
    Run one train epoch
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output = model(input_var)
        loss = (criterion(output, target_var) - config.b).abs() + config.b

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    print('Total time for epoch [{0}] : {1:.3f}'.format(epoch, batch_time.sum))

    wandb.log({"train acc": top1.avg, "train loss": losses.avg, "epoch": epoch})
    arr_time.append(batch_time.sum)


def validate(val_loader, model, criterion, config):
    """
    Run evaluation
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)

if __name__ == '__main__':
    main()