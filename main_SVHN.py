import argparse
import os.path

import matplotlib.pyplot as plt

import dynconv
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader,random_split
import time
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import utils.flopscounter as flopscounter
import utils.logger as logger
import utils.utils as utils
import utils.viz as viz
from torch.backends import cudnn as cudnn
import models

cudnn.benchmark = True

device = 'cuda:1'


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with sparse masks')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=[150,250], nargs='+', type=int, help='learning rate decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--model', type=str, default='resnet32', help='network model name')


    # parser.add_argument('--resnet_n', default=5, type=int, help='number of layers per resnet stage (5 for Resnet-32)')
    parser.add_argument('--budget', default=-1, type=float, help='computational budget (between 0 and 1) (-1 for no sparsity)')
    parser.add_argument('-s', '--save_dir', type=str, default='', help='directory to save model')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluation mode')
    parser.add_argument('--plot_ponder', action='store_true', help='plot ponder cost')
    parser.add_argument('--workers', default=8, type=int, help='number of dataloader workers')
    parser.add_argument('--pretrained', action='store_true', help='initialize with pretrained model')
    args =  parser.parse_args()
    print('Args:', args)


    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# dataloding add a line datset to length 

    dataset = SVHN(root='data/', download=True, transform=transform_train)

    test_size = 12000
    train_size = len(dataset) - test_size

    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    len(train_ds), len(test_ds)


    train_loader = DataLoader(train_ds, batch_size = 256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_ds, batch_size = 256, num_workers=4, pin_memory=True)


    ## MODEL
    net_module = models.__dict__[args.model]
    model = net_module(sparse=args.budget >= 0, pretrained=args.pretrained).to(device=device)

    ## CRITERION
    class Loss(nn.Module):
        def __init__(self):
            super(Loss, self).__init__()
            self.task_loss = nn.CrossEntropyLoss().to(device=device)
            self.sparsity_loss = dynconv.SparsityCriterion(args.budget, args.epochs) if args.budget >= 0 else None

        def forward(self, output, target, meta):
            l = self.task_loss(output, target) 
            logger.add('loss_task', l.item())
            if self.sparsity_loss is not None:
                l += 10*self.sparsity_loss(meta)
            return l
    
    criterion = Loss()

    ## OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    ## CHECKPOINT
    start_epoch = -1
    best_prec1 = 0

    if not args.evaluate and len(args.save_dir) > 0:
        if not os.path.exists(os.path.join(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir))

    if args.resume:
        resume_path = args.resume
        if not os.path.isfile(resume_path):
            resume_path = os.path.join(resume_path, 'checkpoint.pth')
        if os.path.isfile(resume_path):
            print(f"=> loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']-1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{resume_path}'' (epoch {checkpoint['epoch']}, best prec1 {checkpoint['best_prec1']})")
        else:
            msg = "=> no checkpoint found at '{}'".format(resume_path)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)


    try:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=args.lr_decay, last_epoch=start_epoch)
    except:
        print('Warning: Could not reload learning rate scheduler')
    start_epoch += 1
            
    ## Count number of params
    print("* Number of trainable parameters:", utils.count_parameters(model))


    ## EVALUATION
    if args.evaluate:
        print(f"########## Evaluation ##########")
        prec1 = validate(args, val_loader, model, criterion, start_epoch)
        return
        
    ## TRAINING
    for epoch in range(start_epoch, args.epochs):
        print(f"########## Epoch {epoch} ##########")

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(args, train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_prec1': best_prec1,
        }, folder=args.save_dir, is_best=is_best)

        print(f" * Best prec1: {best_prec1}")



def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    Run one train epoch
    """
    model.train()

    # set gumbel temp
    # disable gumbel noise in finetuning stage
    gumbel_temp = 1.0
    gumbel_noise = False if epoch > 0.8*args.epochs else True

    num_step =  len(train_loader)
    total_loss = 0.0
    correct_predictions = 0
    start_time = time.time()

    for input, target in tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5):

        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # compute output
        meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch}
        output, meta = model(input, meta)
        loss = criterion(output, target, meta)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate the number of correct predictions
        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == target).sum().item()

        logger.tick()

    end_time = time.time()
    elapsed_time = end_time - start_time
    accuracy = correct_predictions / (len(train_loader.dataset) * 1.0)
    total_time_minutes = elapsed_time / 60.0

    print(f"Epoch [{epoch}] Loss: {total_loss / num_step:.4f} Accuracy: {accuracy:.4f} Time: {total_time_minutes:.2f} minutes")

def validate(args, val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    top1 = utils.AverageMeter()
    top1 = utils.AverageMeter()
    prec_meter = utils.AverageMeter()
    recall_meter = utils.AverageMeter()
    f1_meter = utils.AverageMeter()

    # switch to evaluate mode
    model = flopscounter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    model.reset_flops_count()

    num_step = len(val_loader)
    with torch.no_grad():
        for input, target in tqdm.tqdm(val_loader, total=num_step, ascii=True, mininterval=5):
            input = input.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            # compute output
            meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch}
            output, meta = model(input, meta)
            output = output.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            top1.update(prec1.item(), input.size(0))
            precision, recall, f1 = precision_recall_f1(output.data, target)
            prec_meter.update(precision, input.size(0))
            recall_meter.update(recall, input.size(0))
            f1_meter.update(f1, input.size(0))

            if args.plot_ponder:
                viz.plot_image(input)
                viz.plot_ponder_cost(meta['masks'])
                viz.plot_masks(meta['masks'])
                plt.show()

    print(f'* Epoch {epoch} - Prec@1 {top1.avg:.3f}')
    print(f'* Epoch {epoch} - Precision {prec_meter.avg:.3f} - Recall {recall_meter.avg:.3f} - F1 {f1_meter.avg:.3f}')

    print(f'* average FLOPS (multiply-accumulates, MACs) per image:  {model.compute_average_flops_cost()[0]/1e6:.6f} MMac')
    model.stop_flops_count()
    return top1.avg

def precision_recall_f1(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    target = target.cpu().numpy()

    num_classes = output.size(1)
    
    true_positive = [0] * num_classes
    false_positive = [0] * num_classes
    false_negative = [0] * num_classes
    true_negative = [0] * num_classes
    
    for i in range(num_classes):
        true_positive[i] = ((pred == i) & (target == i)).sum()
        false_positive[i] = ((pred == i) & (target != i)).sum()
        false_negative[i] = ((pred != i) & (target == i)).sum()
        true_negative[i] = ((pred != i) & (target != i)).sum()

    precision = [tp / (tp + fp) if (tp + fp) != 0 else 0 for tp, fp in zip(true_positive, false_positive)]
    recall = [tp / (tp + fn) if (tp + fn) != 0 else 0 for tp, fn in zip(true_positive, false_negative)]
    f1 = [2 * (p * r) / (p + r) if (p + r) != 0 else 0 for p, r in zip(precision, recall)]
    
    avg_precision = sum(precision) / num_classes
    avg_recall = sum(recall) / num_classes
    avg_f1 = sum(f1) / num_classes
    
    return avg_precision, avg_recall, avg_f1
if __name__ == "__main__":
    main()    
