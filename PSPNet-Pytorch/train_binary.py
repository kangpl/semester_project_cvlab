import datetime
import os
import sys
from math import sqrt
from PIL import Image
import numpy as np
import torchvision.transforms as standard_transforms
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pspnet import pspnet as PSPNet 
from datasets import kitti_binary
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from layers import *

torch.autograd.set_detect_anomaly(True)
ckpt_path = 'ckpt'
exp_name = 'kitti_car'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

args = {
    'gpu': 0,
    'epoches': 100,
    'train_batch_size': 2,
    'lr': 1e-2 / sqrt(16 / 2),
    'lr_decay': 0.9,
    'longer_size': 1200,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'print_freq': 10,
    'val_save_to_img_file': True,
    # 'model_path': 'ckpt/city_pedestrian_focal_pyramid/epoch_3_iter_2344_loss_0.00114_lr_0.0013163738_acc_0.99639.pth'
}

def accuracy(pred, gt):
    # print(type(pred), type(gt), pred.dtype, gt.dtype)
    pred = pred.view(-1).int()
    gt = gt.view(-1).int()

    total = len(gt)
    not_correct = (pred ^ gt).sum().item()
    print("total %d not_correct %d" % (total, not_correct))
    return 1 - not_correct/total

def main():
    net = PSPNet(19)
    net.load_pretrained_model(model_path = './Caffe-PSPNet/pspnet101_cityscapes.caffemodel')
    for param in net.parameters():
        param.requires_grad = False
    net.cbr_final = conv2DBatchNormRelu(4096, 128, 3, 1, 1, False)
    net.dropout = nn.Dropout2d(p=0.1, inplace=True)
    net.classification = nn.Conv2d(128, kitti_binary.num_classes, 1, 1, 0)

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    if len(args['snapshot']) == 0:
        # net.load_state_dict(torch.load(os.path.join(ckpt_path, 'cityscapes (coarse)-psp_net', 'xx.pth')))
        args['best_record'] = {'epoch': 0, 'iter': 0, 'val_loss': 1e10, 'accu': 0}
    else:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))
        split_snapshot = args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args['best_record'] = {'epoch': int(split_snapshot[1]), 'iter': int(split_snapshot[3]),'val_loss': float(split_snapshot[5]), 'accu': float(split_snapshot[7])}
    net.cuda(args['gpu']).train()

    mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(args['longer_size']),
        joint_transforms.RandomRotate(10),
        joint_transforms.RandomHorizontallyFlip()
    ])
    train_input_transform = standard_transforms.Compose([
        extended_transforms.FlipChannels(),
        standard_transforms.ToTensor(),
        standard_transforms.Lambda(lambda x: x.mul_(255)),
        standard_transforms.Normalize(*mean_std)
    ])
    val_input_transform = standard_transforms.Compose([
        extended_transforms.FlipChannels(),
        standard_transforms.ToTensor(),
        standard_transforms.Lambda(lambda x: x.mul_(255)),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    train_set = kitti_binary.KITTI(mode = 'train', joint_transform=train_joint_transform,
                                      transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
    val_set = kitti_binary.KITTI(mode = 'val', transform=val_input_transform,
                                    target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=8, shuffle=False)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full([1], 1.05)).cuda(args['gpu'])

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'], nesterov=True)

    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')

    train(train_loader, net, criterion, optimizer, args, val_loader)


def train(train_loader, net, criterion, optimizer, train_args, val_loader):
    max_iter = train_args['epoches'] * len(train_loader)
    print('max_iter: ', max_iter)
    curr_iter = 0
    for epoch in range(train_args['epoches']):
        train_loss = 0
        # train_aux_loss = AverageMeter()
        # curr_iter = (curr_epoch - 1) * len(train_loader)
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * train_args['lr'] * (1 - float(curr_iter) / max_iter
                                                                      ) ** train_args['lr_decay']
            optimizer.param_groups[1]['lr'] = train_args['lr'] * (1 - float(curr_iter) / max_iter
                                                                  ) ** train_args['lr_decay']

            inputs, gts = data
            gts =gts.float()
            inputs = Variable(inputs).cuda(args['gpu'])
            gts = Variable(gts).cuda(args['gpu'])
            # print("inputs: ", inputs.dtype)
            # print("gts: ", gts.dtype)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, gts)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            curr_iter += 1
            writer.add_scalar('train_loss', loss, curr_iter)
            # writer.add_scalar('train_aux_loss', train_aux_loss.avg, curr_iter)
            writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)

            if (i + 1) % train_args['print_freq'] == 0:
                print('[epoch %d], [iter %d / %d], [train loss %.5f]. [lr %.10f]' % (
                    epoch, i + 1, len(train_loader), train_loss / (i + 1),
                    optimizer.param_groups[1]['lr']))
        if epoch % 2 == 0:
            validate(val_loader, net, criterion, optimizer, epoch, curr_iter, train_args)


def validate(val_loader, net, criterion, optimizer, epoch, iter_num, train_args):
    # the following code is written assuming that batch size is 1
    net.eval()

    val_loss = 0
    pred_all = []
    gts_all = []
    for vi, data in enumerate(val_loader):
        img, gts = data
        gts = gts.float()
        with torch.no_grad():
            img = Variable(img).cuda(args['gpu'])
            gts = Variable(gts).cuda(args['gpu'])

            output = net(img)

            val_loss += criterion(output, gts)
        # print('validating: %d / %d' % (vi + 1, len(val_loader)))
        pred = (nn.Sigmoid()(output) > 0.5).float()
        pred_all.append(pred.data.cpu().numpy())
        gts_all.append(gts.data.cpu().numpy())

    gts_all = np.concatenate(gts_all)
    pred_all = np.concatenate(pred_all)
    accu = accuracy(torch.from_numpy(pred_all), torch.from_numpy(gts_all))

    if val_loss/len(val_loader) < train_args['best_record']['val_loss']:
        train_args['best_record']['val_loss'] = val_loss/len(val_loader)
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['iter'] = iter_num
        train_args['best_record']['accu'] = accu
    snapshot_name = 'epoch_%d_iter_%d_loss_%.5f_lr_%.10f_acc_%.5f' % (
        epoch, iter_num, val_loss/len(val_loader), optimizer.param_groups[1]['lr'], accu)
    torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

    if train_args['val_save_to_img_file']:
        to_save_dir = os.path.join(ckpt_path, exp_name, '%d_%d' % (epoch, iter_num))
        check_mkdir(to_save_dir)

    for idx, data in enumerate(zip(gts_all, pred_all)):
        # print(type(data[0]), type(data[1]), data[0].dtype, data[1].dtype, data[0].shape, data[1].shape)
        gt = Image.fromarray(data[0][0].astype(np.uint8)*255)
        pred = Image.fromarray(data[1][0].astype(np.uint8)*255)
        if train_args['val_save_to_img_file']:
            pred.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
            gt.save(os.path.join(to_save_dir, '%d_gt.png' % idx))

    print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [iter %d], [val loss %.5f], [acc %.5f]' % (epoch, iter_num, val_loss/len(val_loader), accu))

    print('best record: [val loss %.5f], [epoch %d],[iter %d],[acc %.5f]' \
        % (train_args['best_record']['val_loss'], train_args['best_record']['epoch'],train_args['best_record']['iter'], accu))

    print('-----------------------------------------------------------------------------------------------------------')

    writer.add_scalar('val_loss', val_loss/len(val_loader), epoch)
    writer.add_scalar('accu', accu, epoch)

    net.train()

if __name__ == '__main__':
    main()
