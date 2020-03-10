import sys
import os
import argparse
import time
from optparse import OptionParser
import numpy as np
import pdb
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import shutil
from eval import eval_net
from unet import NestedUNet,UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, iou_score
from tqdm import tqdm



####### set directory ###########
##### 1. for tensorboard ###
directory = 'runs/avoidDump' 
ct = time.localtime(time.time())
directory = os.path.join(directory, "%04d-%02d-%02d, %02d:%02d:%02d_bce+dice/" %
                                                (ct.tm_year, ct.tm_mon, ct.tm_mday, ct.tm_hour, ct.tm_min, ct.tm_sec))
if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter(directory)

#### 2. save model #####
dir_model = 'fullmodel/avoidDump'
if not os.path.exists(dir_model):
    os.makedirs(dir_model)

best_dice = 0
best_loss = 10
##################################



def train_net(args,
              net,
              epochs,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    global best_dice, best_loss
    dir_img = '/home/mori/Programming/Net_Pruning/unetdataset_patchImg/img/'
    dir_mask = '/home/mori/Programming/Net_Pruning/unetdataset_patchImg/graylabel/'


    ids = get_ids(dir_img)  # get file name (without .png)
    print("ids:{}".format(ids))
    
    ids = split_ids(ids)  # 重采样？
    print("ids:{}".format(ids))
    iddataset = split_train_val(ids, val_percent) # 按给定比例划分打乱的数据集

    ###### count parameters  ############
    paras = sum([p.data.nelement() for p in net.parameters()]) 

    print('''
    Starting training:
        Epochs: {}
        Parameters: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
        Deepsupervision: {}
    '''.format(epochs, paras, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu),str(args.deepsupervision)))

    N_train = len(iddataset['train'])
    print("N_train:{}".format(N_train))
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        New_lr=adjust_learning_rate(optimizer, epoch,epochs)
        print(' lr: {}'.format(New_lr))
        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        best_iou = 0
        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):  # 手动分出batch
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            true_masks_flat = true_masks.view(-1)
            true_masks_flat = true_masks_flat/255  # 归一化
            
            output = net(imgs)
            masks_pred = F.sigmoid(output)

            if args.deepsupervision:
                #### unet++ with deepsupervision
                loss = 0
                for mp in masks_pred:
                    masks_probs_flat = mp.view(-1)
                    loss += criterion(masks_probs_flat, true_masks_flat)
                loss /= len(masks_pred)
                epoch_loss += loss.item()
            else:   
                masks_probs_flat = masks_pred.view(-1)
                loss = criterion(masks_probs_flat, true_masks_flat)
                epoch_loss += loss.item()

                ## todo: adjust iou 
                iou = iou_score(output, true_masks/255)

            ######## record the best iou
            if iou > best_iou:
                best_iou = iou

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
            
            newloss=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
        print('Best iou: {}'.format(best_iou))
        val_dice = eval_net(net, val, gpu)
        print('Validation Dice Coeff: {}'.format(val_dice))

        writer.add_scalar('train_loss',epoch_loss/i,(epoch+1))
        writer.add_scalar('val_dice', val_dice, (epoch+1))
        writer.add_scalar('best iou', best_iou, (epoch+1))


        if save_cp:
            #torch.save(net.state_dict(),dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            #print('Checkpoint {} saved !'.format(epoch + 1))
            dice_best = val_dice > best_dice
            loss_best = epoch_loss / i < best_loss
            best_dice = max(val_dice, best_dice)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_dice': best_dice,
                'best_loss': best_loss, 
            }, dice_best, loss_best)
        
    print('Best dice: ', best_dice)

    
def adjust_learning_rate(optimizer, epoch,epochs):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    # lr = args.lr * (0.1 ** (epoch // (20+epochs*0.1+0.12*epoch))*(4**((epoch // (20+epochs*0.1+0.12*epoch))//3)))
    if epoch<=int(epochs*0.2):
        lr = args.lr
    elif epoch>int(epochs*0.2) and epoch<=int(epochs*0.35):
        lr = args.lr*0.1*3
    elif epoch>int(epochs*0.35) and epoch<=int(epochs*0.5):
        lr = args.lr*0.1
    elif epoch>int(epochs*0.5) and epoch<=int(epochs*0.65):
        lr = args.lr*0.01*3
    elif epoch>int(epochs*0.65) and epoch<=int(epochs*0.8):
        lr = args.lr*0.01
    elif epoch>int(epochs*0.8) and epoch<=int(epochs*0.95):
        lr = args.lr*0.001*3
    elif epoch>int(epochs*0.95):
        lr = args.lr*0.001

    writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, dice_best, loss_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    filename = directory + filename
    torch.save(state, filename)
    if dice_best:
        shutil.copyfile(filename,os.path.join(directory, 'dice_best.pth'))
    if loss_best:
        shutil.copyfile(filename,os.path.join(directory, 'loss_best.pth'))

'''args for not official Unet++  '''
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=6,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('--tensorboard', default=True,
                    help='Log progress to TensorBoard', action='store_true')
    ##### for official unet++
    parser.add_option('--input-channels', default=3, type='int',
                         help='input channels')
    parser.add_option('-d','--deepsupervision', default=0)   
    ##############
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)
    # net = NestedUNet(args)
    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory


    ######## model visualization in tensorboard ##############
    dummy_input = torch.rand(1,3,256,256).cuda()
    writer.add_graph(net,(dummy_input,))
    #########################################################

    
    if args.load:
        if os.path.isfile(args.load):
            print("=> loading checkpoint '{}'".format(args.load))
            checkpoint = torch.load(args.load)
            args.start_epoch = checkpoint['epoch']
            best_dice = checkpoint['best_dice']
            best_loss = checkpoint['best_loss']
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.load, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.load))

    try:
        train_net(args = args,
                  net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)

        torch.save(net,
                    dir_model + 'CP{}.pth'.format('Unet+_200epoch'))
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
    # writer.close()