#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import time
import shutil
import os
import csv
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

# Add the root directory to the path so we can import modules properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.ske_mixf import Model
from eval.eval_loader import Dataloaders, AverageMeter
from data import datasets  # Import datasets for correct number of classes

# Set random seed for reproducibility
np.random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

def accuracy(output, target):
    """Compute the accuracy for a batch of predictions."""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)
    return correct.mul_(100.0 / batch_size).item()

def validate(data_loader, model, device, criterion, print_freq=10):
    """Run model validation on the validation set."""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            input = input.to(device)
            target = target.to(device)

            # Reshape input to match the expected format for Mixformer
            # Current shape is (N, T, F) where F=75 (25 joints * 3 channels for first actor)
            # Need to reshape to (N, C, T, V, M) where M=2 (number of people)
            N, T, F = input.size()
            C = 3  # x, y, z coordinates
            V = 25  # 25 joints
            M = 2  # 2 people

            # Check if we have the expected number of frames (64)
            expected_frames = 64
            if T < expected_frames:
                # Pad with zeros if we don't have enough frames
                padded_input = torch.zeros(N, expected_frames, F, device=input.device)
                padded_input[:, :T, :] = input
                input = padded_input
                T = expected_frames
            elif T > expected_frames:
                # Truncate if we have too many frames
                input = input[:, :expected_frames, :]
                T = expected_frames

            # First reshape to (N, T, V, C) for the first actor
            input_reshaped = input.view(N, T, V, C)

            # Create a tensor for both actors (second actor is all zeros)
            input_with_two_actors = torch.zeros(N, T, V, C, M, device=input.device)
            input_with_two_actors[:, :, :, :, 0] = input_reshaped  # First actor

            # Permute to (N, C, T, V, M)
            input = input_with_two_actors.permute(0, 3, 1, 2, 4).contiguous()

            # Forward pass
            output = model(input)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(data_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

    print(' * Accuracy {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, losses.avg

def train(data_loader, model, device, criterion, optimizer, epoch, print_freq=10):
    """Train the model for one epoch."""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move data to the target device
        input = input.to(device)
        target = target.to(device)

        # Reshape input to match the expected format for Mixformer
        # Current shape is (N, T, F) where F=75 (25 joints * 3 channels for first actor)
        # Need to reshape to (N, C, T, V, M) where M=2 (number of people)
        N, T, F = input.size()
        C = 3  # x, y, z coordinates
        V = 25  # 25 joints
        M = 2  # 2 people

        # Check if we have the expected number of frames (64)
        expected_frames = 64
        if T < expected_frames:
            # Pad with zeros if we don't have enough frames
            padded_input = torch.zeros(N, expected_frames, F, device=input.device)
            padded_input[:, :T, :] = input
            input = padded_input
            T = expected_frames
        elif T > expected_frames:
            # Truncate if we have too many frames
            input = input[:, :expected_frames, :]
            T = expected_frames

        # First reshape to (N, T, V, C) for the first actor
        input_reshaped = input.view(N, T, V, C)

        # Create a tensor for both actors (second actor is all zeros)
        input_with_two_actors = torch.zeros(N, T, V, C, M, device=input.device)
        input_with_two_actors[:, :, :, :, 0] = input_reshaped  # First actor

        # Permute to (N, C, T, V, M)
        input = input_with_two_actors.permute(0, 3, 1, 2, 4).contiguous()

        # Forward pass
        output = model(input)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        acc1 = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

    return top1.avg, losses.avg

def main():
    parser = argparse.ArgumentParser(description='Skeleton MixFormer Training')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='NTU',
                        help='Dataset to use (NTU, NTU120, ETRI)')
    parser.add_argument('--case', type=int, default=0,
                        help='0: Cross-Subject, 1: Cross-View, 2: Cross-Setup (NTU120 only)')
    parser.add_argument('--tag', type=str, default='ar',
                        help='Task: ar (action recognition), ri (re-identification), or gc (gender classification)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--seg', type=int, default=20,
                        help='Number of frames per segment sequence')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--lr_decay_interval', type=int, default=30,
                        help='Learning rate decay interval (epochs)')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='Learning rate decay factor')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')

    # Checkpoint and output parameters
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for models and logs')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint to resume training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the model without training')

    args = parser.parse_args()

    # Set up output directory
    split_tags = {0: 'csub', 1: 'cview', 2: 'cset'}
    split_tag = split_tags[args.case]
    model_dir = f'{args.dataset}_mixformer_{args.tag}_{split_tag}'
    save_path = os.path.join(args.output_dir, model_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Map the dataset name to our internal dataset keys
    dataset_mapping = {
        'NTU': 'ntu',
        'NTU120': 'ntu120',
        'ETRI': 'etri'
    }
    dataset_key = dataset_mapping.get(args.dataset, args.dataset.lower())

    # Determine the number of classes based on dataset and task
    if args.tag == 'ar':  # Action recognition
        num_classes = datasets[dataset_key]['num_class']
    elif args.tag == 'ri':  # Re-identification
        num_classes = datasets[dataset_key]['num_actor']
    elif args.tag == 'gc':  # Gender classification (binary)
        num_classes = 2

    print(f"Creating MixFormer model with {num_classes} classes for {args.dataset} dataset")

    # Load data loaders
    print("Loading data...")
    data_loaders = Dataloaders(
        dataset=args.dataset,
        case=args.case,
        seg=args.seg,
        tag=args.tag
    )

    train_loader = data_loaders.get_train_loader(args.batch_size, args.workers)
    val_loader = data_loaders.get_val_loader(args.batch_size, args.workers)
    test_loader = data_loaders.get_test_loader(args.batch_size, args.workers)

    train_size = data_loaders.get_train_size()
    val_size = data_loaders.get_val_size()

    print(f'Training on {train_size} samples')
    print(f'Validating on {val_size} samples')

    # Create the model
    graph_name = datasets[dataset_key]['graph']
    graph_args = datasets[dataset_key]['graph_args']
    model = Model(
        num_class=num_classes,
        num_point=25,
        num_person=2,
        graph=graph_name,
        graph_args=graph_args
    )
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    # Define learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[args.lr_decay_interval, args.lr_decay_interval * 2],
        gamma=args.lr_factor
    )

    # Load checkpoint if provided
    best_acc = 0
    start_epoch = 0
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint '{args.checkpoint}'")
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            print(f"Loaded checkpoint (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.checkpoint}'")

    # CSV logger setup
    csv_path = os.path.join(save_path, 'log.csv')
    csv_fields = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']

    if not args.evaluate and (not os.path.exists(csv_path) or start_epoch == 0):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

    # Evaluate only
    if args.evaluate:
        print("Evaluating the model...")
        val_acc, val_loss = validate(test_loader, model, device, criterion, args.print_freq)
        print(f"Validation accuracy: {val_acc:.2f}%")
        return

    # Training loop
    for epoch in range(start_epoch, args.max_epochs):
        # Train for one epoch
        train_acc, train_loss = train(
            train_loader, model, device, criterion, optimizer, epoch, args.print_freq
        )

        # Evaluate on validation set
        val_acc, val_loss = validate(
            val_loader, model, device, criterion, args.print_freq
        )

        # Adjust learning rate
        scheduler.step()

        # Log metrics
        log_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(log_data)

        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_path, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copyfile(
                os.path.join(save_path, 'checkpoint.pth.tar'),
                os.path.join(save_path, 'model_best.pth.tar')
            )
            print(f"New best model saved with accuracy {best_acc:.3f}")

    # Final evaluation on the test set
    print("Final evaluation on test set:")
    validate(test_loader, model, device, criterion, args.print_freq)

if __name__ == '__main__':
    main()
