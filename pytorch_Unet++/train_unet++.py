import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
dir_checkpoint = 'C:\\Users\\fs\\Desktop\\pytorch_Unet2\\checkpoints\\'
best_prec1=2
best_val_dice=0
def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):


    dir_img = 'F:\\shunmei\\0784_new\\little_image\\'
    dir_mask = 'F:\\shunmei\\0784_new\\little_label\\'


    ids = get_ids(dir_img)
    print("ids:{}".format(ids))
    
    ids = split_ids(ids)
    print("ids:{}".format(ids))
    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])
    print("N_trian:{}".format(N_train))
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

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
            newloss=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, gpu)
            global best_val_dice
            is_best = best_val_dice < val_dice
            best_val_dice = max(val_dice, best_val_dice)
            if is_best:
                torch.save(net.state_dict(), dir_checkpoint+'M_best.pth')
                torch.save(net, dir_checkpoint+'0_M_best.pth')
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            global best_prec1
            is_best = newloss < best_prec1
            best_prec1 = min(newloss, best_prec1)
            if is_best:
                torch.save(net.state_dict(), dir_checkpoint+'best.pth')
                torch.save(net, dir_checkpoint+'0best.pth')
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))

            print('Checkpoint {} saved !'.format(epoch + 1))
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

    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
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
    parser.add_option('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)

        torch.save(net,
                    dir_checkpoint + 'CP{}.pth'.format(0))
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)