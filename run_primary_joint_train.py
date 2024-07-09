#-------------------------------------------------------------------------------
# Name:        run_joint_finetune.py
# Purpose:     Finetuning regression and attention modules togather with a meanshift module
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
sys.path.append("./")
import os
import numpy as np
import shutil
import argparse
import csv
import json

from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.io_utils import output_point_cloud_ply

import torch
import torch.backends.cudnn as cudnn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.GCN import TARigPrimaryJointPredNet
from datasets.skeleton_dataset import GraphDataset

       
USE_NORMALS = True
ATTENTION_WEIGHTS_ACTIVATIONS = 'softmax' # 'softmax' or 'sigmoid'
JOINT_POSITION_ERROR_LOSS = 'sse' # 'mse' or 'sse'
JOINT_POSITION_ERROR_LOSS_WEIGHT = 1.0
ATTENTION_MAP_LOSS = True
ATTENTION_MAP_LOSS_WEIGHT = 1.0
CHAMFER_LOSS_INBETWEEN_JOINTS = True
CHAMFER_LOSS_INBETWEEN_JOINTS_WEIGHT = 1.0
CHAMFER_LOSS_BETWEEN_JOINTS_AND_VERTICES = False
CHAMFER_LOSS_BETWEEN_JOINTS_AND_VERTICES_WEIGHT = 1.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def write_to_csv(csv_path, err_by_joint):
    header = [str(i) for i in range(len(err_by_joint))]
    data = [err_by_joint[i] for i in range(len(err_by_joint))]
    with open(csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)
        f.close()


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def main(args):
    global device
    lowest_loss = 1e20

    # create checkpoint dir and log dir
    if not isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
    mkdir_p(args.checkpoint)
    if not args.resume:
        if isdir(args.logdir):
            shutil.rmtree(args.logdir)
        mkdir_p(args.logdir)

    # create model
    model = TARigPrimaryJointPredNet(input_normal=USE_NORMALS, num_joints=args.num_joints, attention_weights_activation=ATTENTION_WEIGHTS_ACTIVATIONS,aggr='max')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = DataLoader(GraphDataset(root=args.train_folder), batch_size=args.train_batch, shuffle=not args.no_shuffle, follow_batch=['joints'])
    val_loader = DataLoader(GraphDataset(root=args.val_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])
    test_loader = DataLoader(GraphDataset(root=args.test_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_mae, test_mae_by_joint = test(test_loader, model, args, save_result=True, best_epoch=args.start_epoch)
        write_to_csv(os.path.join(args.checkpoint, f'err_by_joint_eval_test_epoch{args.start_epoch}.csv'), test_mae_by_joint)
        print('test_loss {:8f}'.format(test_loss))
        print('test_mae {:8f}'.format(test_mae))
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.5)
    logger = SummaryWriter(log_dir=args.logdir)
    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch: %d ' % (epoch + 1))
        train_loss, train_mae = train(train_loader, model, optimizer, args)
        val_loss, val_mae, val_mae_by_joint = test(val_loader, model, args)
        target_dir = os.path.join(args.checkpoint, 'csv')
        os.makedirs(target_dir, exist_ok=True)
        write_to_csv(os.path.join(target_dir, f'err_by_joint_val_epoch{epoch}.csv'), val_mae_by_joint)
        test_loss, test_mae, test_mae_by_joint = test(test_loader, model, args)
        write_to_csv(os.path.join(target_dir, f'err_by_joint_test_epoch{epoch}.csv'), test_mae_by_joint)
        print('Epoch{:d}. train_loss: {:.9f}.'.format(epoch + 1, train_loss))
        print('Epoch{:d}.  train_mae: {:.9f}.'.format(epoch + 1, train_mae))
        print('Epoch{:d}.   val_loss: {:.9f}.'.format(epoch + 1, val_loss))
        print("Epoch{:d}.    val_mae: {:.9f}.".format(epoch + 1, val_mae))
        print('Epoch{:d}.  test_loss: {:.9f}.'.format(epoch + 1, test_loss))
        print("Epoch{:d}.   test_mae: {:.9f}.".format(epoch + 1, test_mae))

        # remember best acc and save checkpoint
        is_best = val_loss < lowest_loss
        lowest_loss = min(val_loss, lowest_loss)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lowest_loss': lowest_loss, 'optimizer': optimizer.state_dict()},
                        is_best, checkpoint=args.checkpoint)

        info = {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch+1)

        scheduler.step(val_loss)

    print("=> loading checkpoint '{}'".format(os.path.join(args.checkpoint, 'model_best.pth.tar')))
    checkpoint = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
    best_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(os.path.join(args.checkpoint, 'model_best.pth.tar'), best_epoch))
    test_loss, test_mae, test_mae_by_joint = test(test_loader, model, args, save_result=True, best_epoch=best_epoch)
    write_to_csv(os.path.join(args.checkpoint, f'err_by_joint_test_best_epoch{best_epoch}.csv'), test_mae_by_joint)
    print('Best epoch:\n test_loss {:8f}'.format(test_loss))
    print('Best epoch:\n test_mae {:8f}'.format(test_mae))

def compute_individual_loss(pred, gt, vertices=None, gt_attention_map=None, pred_attention_map=None):
    loss = 0.0

    # Attention map loss
    if ATTENTION_MAP_LOSS:
        # gt_attention_map is a binary mask of shape (num_vertices, num_joints)
        # pred_attention_map is a softmax of shape (num_vertices, num_joints)
        # We employ Dice Loss.
        loss_attention = 1.0 - (2.0 * torch.sum(gt_attention_map * pred_attention_map)) / (torch.sum(gt_attention_map) + torch.sum(pred_attention_map) + 1e-9)
        loss += ATTENTION_MAP_LOSS_WEIGHT * loss_attention

    # Joint position error loss 
    losses_by_joint_component = ((pred - gt) ** 2)
    losses_by_joint = losses_by_joint_component.sum(dim=1)

    if JOINT_POSITION_ERROR_LOSS == 'mse':
        mse_loss = torch.mean(losses_by_joint)
        joint_position_error_loss = mse_loss
    elif JOINT_POSITION_ERROR_LOSS == 'sse':
        sse_loss = torch.sum(losses_by_joint_component)
        joint_position_error_loss = sse_loss
    else:
        raise ValueError(f'Unknown joint position error loss type: {JOINT_POSITION_ERROR_LOSS}')
    loss += JOINT_POSITION_ERROR_LOSS_WEIGHT * joint_position_error_loss

    # Chamfer loss inbetween joints
    if CHAMFER_LOSS_INBETWEEN_JOINTS:
        chamfer_loss_1 = torch.mean(torch.min(torch.sum((pred.unsqueeze(1) - gt.unsqueeze(0)) ** 2, dim=2), dim=1)[0])
        chamfer_loss_2 = torch.mean(torch.min(torch.sum((gt.unsqueeze(1) - pred.unsqueeze(0)) ** 2, dim=2), dim=1)[0])
        chamfer_loss = chamfer_loss_1 + chamfer_loss_2
        loss += CHAMFER_LOSS_INBETWEEN_JOINTS_WEIGHT * chamfer_loss

    if CHAMFER_LOSS_BETWEEN_JOINTS_AND_VERTICES:
        assert vertices is not None
        chamfer_loss_1 = torch.mean(torch.min(torch.sum((pred.unsqueeze(1) - vertices.unsqueeze(0)) ** 2, dim=2), dim=1)[0])
        chamfer_loss_2 = torch.mean(torch.min(torch.sum((vertices.unsqueeze(1) - pred.unsqueeze(0)) ** 2, dim=2), dim=1)[0])
        chamfer_loss = chamfer_loss_1 + chamfer_loss_2
        loss += CHAMFER_LOSS_BETWEEN_JOINTS_AND_VERTICES_WEIGHT * chamfer_loss

    # MAE is the appropriate metric for MPJPE.
    abs_err_by_joint = torch.sqrt(losses_by_joint)
    return loss, abs_err_by_joint

def train(train_loader, model, optimizer, args):
    global device
    model.train()  # switch to train mode
    loss_meter = AverageMeter()
    total_abs_err_by_joint = torch.zeros(args.num_joints, device=device, requires_grad=False)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_pred, attention_weights = model(data)
        loss_total = 0.0
        for i in range(len(torch.unique(data.batch))):
            joint_gt = data.joints[data.joints_batch == i, :]
            individual_loss, individual_abs_err_by_joint = compute_individual_loss(y_pred[i], joint_gt, vertices=data.pos[data.batch == i], gt_attention_map=data.mask[data.batch == i, :], pred_attention_map=attention_weights[i])
            loss_total += individual_loss
            total_abs_err_by_joint += individual_abs_err_by_joint
        loss_total /= len(torch.unique(data.batch))
        loss_total.backward()
        total_abs_err_by_joint /= len(torch.unique(data.batch))
        optimizer.step()
        loss_meter.update(loss_total.item())
    total_abs_err_by_joint /= len(train_loader.dataset)
    mae = torch.mean(total_abs_err_by_joint)
    return loss_meter.avg, mae

def test(test_loader, model, args, save_result=False, best_epoch=None):
    global device
    model.eval()  # switch to test mode
    loss_meter = AverageMeter()
    outdir = args.checkpoint.split('/')[-1]
    total_abs_err_by_joint = None
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            y_pred, attention_weights = model(data)
            loss_total = 0.0
            batch_size = len(torch.unique(data.batch))
            for i in range(batch_size):
                joint_gt = data.joints[data.joints_batch == i, :]
                individual_loss, individual_abs_err_by_joint = compute_individual_loss(y_pred[i], joint_gt, vertices=data.pos[data.batch == i], gt_attention_map=data.mask[data.batch == i, :], pred_attention_map=attention_weights[i])
                loss_total += individual_loss
                if total_abs_err_by_joint is None:
                    total_abs_err_by_joint = torch.zeros_like(individual_abs_err_by_joint)
                total_abs_err_by_joint += individual_abs_err_by_joint
                if save_result:
                    output_point_cloud_ply(y_pred[i], name=data.name[i],
                                           output_folder=f'{outdir}/results/best_{best_epoch}/')
                    # dump json of y_pred[i]
                    with open(f'{outdir}/results/best_{best_epoch}/{data.name[i]}.json', 'w') as f:
                        json.dump(y_pred[i].tolist(), f)
                                        
            loss_total /= batch_size
            loss_meter.update(loss_total.item())
    total_abs_err_by_joint /= len(test_loader.dataset)
    mae = torch.mean(total_abs_err_by_joint)
    total_abs_err_by_joint = total_abs_err_by_joint.tolist()
    return loss_meter.avg, mae, total_abs_err_by_joint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyG DGCNN')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on val/test set')
    parser.add_argument('--train_batch', default=1, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--test_batch', default=1, type=int, metavar='N', help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--logdir', default='logs/test', type=str, metavar='LOG', help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_folder', required=True, type=str, help='folder of training data')
    parser.add_argument('--val_folder', required=True, type=str, help='folder of validation data')
    parser.add_argument('--test_folder', required=True, type=str, help='folder of testing data')
    parser.add_argument('--no-shuffle', action='store_true', help='no shuffle')
    parser.add_argument('--lr', default=1e-3, type=float) # Try 5e-5
    parser.add_argument('--num-joints', type=int, required=True, help='Number of joints')
    print(parser.parse_args())
    main(parser.parse_args())
