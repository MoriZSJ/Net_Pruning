import cv2
import os
from tqdm import tqdm

oldpath='/home/mori/Programming/Net_Pruning/densenet-pytorch-master/0770/unet/ok&&NG/20190308bmp0010'
newpath=oldpath+'_new/'

os.chdir(oldpath)
if not os.path.exists(newpath):
    os.makedirs(newpath)
for img in tqdm(os.listdir()):
    if img.endswith('bmp'):
        try:
            i=cv2.imread(img)
            o=cv2.imwrite(newpath+img,i)
        except:
            print(img)


