import sys
sys.path.append("/home/mori/Programming/Net_Pruning/densenet-pytorch-master")
print(sys.path)
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont
import pdb
from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks
# from crf import dense_crf
from utils import plot_img_and_mask
import pdb
from torchvision import transforms
import densenet as dn
dirs = '/home/mori/Programming/Net_Pruning/densenet-pytorch-master/test_image_results/'   # test img folder
dense12 = '/home/mori/Programming/Net_Pruning/densenet-pytorch-master/runs/DenseNet_focal_772_2_12/model_best.pth'
# dense24 = 'C:\\Users\\fs\\Desktop\\densenet-pytorch-master\\runs_k=24\\DenseNet_Unet_fs\\model_best.pth'
files = os.listdir(dirs)
for index,value in enumerate(files):
    files[index] = dirs + files[index]
def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_gpu=False):
    pst = time.time()
    net.eval()

    #print(' 0 running time: %s seconds ' %(( time.clock() -pst)))


    img_height = full_img.size[1]
    print(img_height)
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    #pdb.set_trace()
    #print(' 1 running time: %s seconds ' %(( time.clock() -pst)))

    img = normalize(img)

    #print(' 2 running time: %s seconds ' %(( time.clock() -pst)))

    left_square, right_square = split_img_into_squares(img)

    left_square = hwc_to_chw(left_square)
    right_square = hwc_to_chw(right_square)

    #print(' 3 running time: %s seconds ' %(( time.clock() -pst)))


    X_left = torch.from_numpy(left_square).unsqueeze(0)
    X_right = torch.from_numpy(right_square).unsqueeze(0)

    #print(' 4 running time: %s seconds ' %(( time.clock() -pst)))

    #outstart = time.clock()
    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()

    #print(' 5 running time: %s seconds ' %(( time.clock() -pst)))


    with torch.no_grad():
        torch.cuda.synchronize()
        st = time.time()
        output_left = net(X_left)
        output_right = net(X_right)
        torch.cuda.synchronize()
        st1 = time.time()
        end = st1 - st
        #outend = time.clock()
        print(' --------------------unet running time: %s seconds ' %(end))
        left_probs = output_left.squeeze(0)
        right_probs = output_right.squeeze(0)
        print(' squeeze running time: %s seconds ' %(( time.time() -pst)))

        if (left_probs.shape[1] != img_height) :
            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(img_height),
                    transforms.ToTensor()
                ]
            )
            left_probs = tf(left_probs.cpu())
            right_probs = tf(right_probs.cpu())
            print("11111")
        #print(' 8running time: %s seconds ' %(( time.clock() -pst)))
        #lstart = time.clock()
        
        #left_probs.cpu()
        #print(' transforms running time: %s seconds ' %(( time.time() -pst)))
        st = time.time()
        left_mask_np = left_probs.squeeze().cpu().numpy()
        end1 = time.time() - st
        #print(left_probs.shape)
        #pdb.set_trace()
        print(' tonumpy1 running time: %s seconds ' %(end1))
        st = time.time()
        right_mask_np = right_probs.squeeze().cpu().numpy()
        end2 = time.time() - st
        print(' tonumpy2 running time: %s seconds ' %(end2))
    #pdb.set_trace()
    full_mask = merge_masks(left_mask_np, right_mask_np, img_width)
    #print(type(full_mask))
    
    #print((full_mask.size))
    print(' 9 running time: %s seconds ' %(( time.time() -pst)))

            #pdb.set_trace()
    full_mask[full_mask >= out_threshold] = 1
    full_mask[full_mask < out_threshold] = 0
        #-------------------------------------------------------------------------------

        #newmask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)
    #lend = time.clock()
    #print(' running time: %s seconds ' %((lend-pst)))
    #pdb.set_trace()
    return full_mask > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='/home/mori/Programming/Net_Pruning/pytorch_Unet++/runs/2019-12-05, 15:15:28_bce+dice/dice_best.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=False)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--usedense', '-r', action='store_true',
                        help=" use densenet k=12(False to k=24)",
                        default=True)
    parser.add_argument('--best', '-b', action='store_true', dest = 'best',
                        help=" use densenet k=12(False to k=24)",
                        default=True)
    parser.add_argument('--describe', '-d', action='store_true',
                        help=" whether add the describle text into mask",
                        default=True)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)  # test-327: 1
    parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
    parser.set_defaults(bottleneck=True)   
    return parser.parse_args()
    # return parser.parse_args(["-i","1.jpg"])

def get_output_filenames(args):
    in_files = files
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def get_describe_filenames(args):
    in_files = files
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_DES{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files
def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))




if __name__ == "__main__":
    args = get_args()
    in_files = files
    out_files = get_output_filenames(args)

    if args.describe:
        des_files = get_describe_filenames(args)
        print("Will output images with descriptors !")
        patchscale = 100
        dense_normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])


    net = UNet(n_channels=3, n_classes=1)
    
    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        if args.best:
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.model, checkpoint['epoch']))
        else:
            print("=> not best model ")
            net.load_state_dict(torch.load(args.model))

    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    densenet = dn.DenseNet3(16, 3, args.growth, bottleneck=args.bottleneck,  small_inputs = False)
    densenet = densenet.cuda()

    if args.usedense:
        if args.growth == 12:
            if os.path.isfile(dense12):
                print("=> loading checkpoint '{}'".format(dense12))
                checkpoint = torch.load(dense12)
                args.start_epoch = checkpoint['epoch']
                #best_prec1 = checkpoint['best_prec1']
                densenet.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                        .format(dense12, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(dense12))
        elif  args.growth == 24:
            if os.path.isfile(dense12):
                print("=> loading checkpoint '{}'".format(dense24))
                checkpoint = torch.load(dense24)
                args.start_epoch = checkpoint['epoch']
                #best_prec1 = checkpoint['best_prec1']
                densenet.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                        .format(dense24, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(dense24))

    print("Model loaded !")

    classes = []
    with open('classes.txt', 'r') as list_:
        for line in list_:
            classes.append(line.rstrip('\n'))

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
        start = time.clock()
        img = Image.open(fn)
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")
        if img.mode != 'RGB':
            img = img.convert('RGB')
        width = img.size[0]
        height = img.size[1]
        #box1 = (0,int(height*2/5),width,height)
        box1 = (0,int(height*1/5),width,height)
        #box2 = (width/2,int(height*2/5),width,height)
        cimg1 = img.crop(box1)
        #cimg2 = img.crop(box2)
        mask = np.zeros((int(height),int(width)), np.uint8)
        a = torch.rand([5,5]).float().cuda()
        # s1 = time.time()
        # a.cpu()
        # s2 = time.time()
        #print("-------------------------------------------------------%s" %(s2-s1))
        tmpmask = predict_img(net=net,
                           full_img=cimg1,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_gpu=not args.cpu)
        mask[int(height*1/5):int(height),0:int(width)] = tmpmask

        print(' unet ------------------------------->running time: %s seconds ' %(( time.clock() -start)))
        #tmp = time.clock()
        # mask2 = predict_img(net=net,
        #                    full_img=cimg2,
        #                    scale_factor=args.scale,
        #                    out_threshold=args.mask_threshold,
        #                    use_densenet= not args.no_crf,
        #                    use_gpu=not args.cpu)


        # gray = cv2.cvtColor(full_mask,cv2.COLOR_BGR2GRAY)  
        # gray = full_mask
        # ret, binary = cv2.threshold(gray,15,255,cv2.THRESH_BINARY)  
        # binary = np.clip(binary, 0, 255)# 归一化也行
        # binary = np.array(binary,np.uint8)
        

        #---------------------------------patch_crf_function
        if args.usedense:
            densenet.eval()
            contours, hierarchy = cv2.findContours(np.array(mask).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
            print('The whole suspicious contours: %s' %(len(contours)))
            print(' findContours ------------------------------->running time: %s seconds ' %(( time.clock() -start)))
            classnum = [0,0,0]
            #pdb.set_trace()
            if args.describe:
                desmask = Image.fromarray((mask * 255).astype(np.uint8))
                if desmask.mode != 'RGB':
                    desmask = desmask.convert('RGB')
                draw = ImageDraw.Draw(desmask)
                ttfront = ImageFont.load_default()#字体大小
            
            if (len(contours) != 0):
                densemask = np.zeros((int(mask.shape[0]),int(mask.shape[1])), np.uint8)
                
                for k in range(len(contours)):
                    x, y, wid, hei = cv2.boundingRect( np.array(contours[k]))
                    # if (wid < 10 and hei < 10):
                    #     continue
                    #pdb.set_trace()
                    midx = int(x + wid/2)
                    midy = int(y + hei/2)
                    midx = midx if (midx >= patchscale and midx <= (mask.shape[1] - patchscale)) else ( patchscale  if (midx < patchscale) else (mask.shape[1] - patchscale))
                    midy = midy if (midy >= patchscale and midy <= (mask.shape[0] - patchscale)) else ( patchscale  if (midy < patchscale) else (mask.shape[0] - patchscale))

                    if (densemask[midy,midx] == 1):
                        continue
                

                    
                    patch_img = img.crop((midx-patchscale,midy-patchscale,midx+patchscale,midy+patchscale))
                    # patch_mask = mask[(midy-patchscale):(midy+patchscale),(midx-patchscale):(midx+patchscale)]
                    # patch_mask1 = mask_to_image(patch_mask)
                    # patch_mask1.save("num{}_mask.png".format(k))
                    # patch_img.save("num{}.png".format(k))
                    transform_i = transforms.Compose([    
                                transforms.ToTensor(),
                                dense_normalize,
                                    ])
                    input = transform_i(patch_img)
                    input = input.unsqueeze(0)
                    input = input.cuda()
                    input = torch.autograd.Variable(input)
                    #print(input)
                    #pdb.set_trace()
                    torch.cuda.synchronize()
                    dens = time.time()
                    output = densenet(input)
                    torch.cuda.synchronize()
                    dene = time.time()
                    dentime = dene - dens
                    print(' densenet ------------------------------->running time: %s seconds ' %(( dentime)))
                    #print(output)
                    x= torch.max(output,1)
                    classnum[x[1]] += 1
                    
                    if args.describe:
                        draw.text((midx, midy),str(classes[x[1]]),fill=(255,0,0), font=ttfront)#文字位置，内容，字体
                    # if (classnum[1] >= 3):
                    #     break

                    #print('Prediction: ', str(classes[x[1]]))
                    densemask[midy-patchscale:midy+patchscale, midx-patchscale:midx+patchscale] = 1
            
            print('flaw area: %s ,  bubble area:  %s ,  other area: %s ' %(classnum[1],classnum[0],classnum[2]))

        
        #mask[int(height*2/5):int(height),int(width/2):int(width)] = mask2
        end = time.clock()
        print('Total running time: %s Seconds'  %((end-start)))
        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
        
        #print('sub running time: %s seconds and %s seconds' %((tmp-start),(end-tmp)))
        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])
            if args.describe:
                desmask.save(des_files[i])
                

            print("Mask saved to {}".format(out_files[i]))
