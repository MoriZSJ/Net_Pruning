import cv2
import os
import sys
from tqdm import tqdm

oldpath='/home/mori/Programming/unet++_official/input/circuit/masks'
newpath=oldpath+'_new/'

os.chdir(oldpath)
if not os.path.exists(newpath):
    os.makedirs(newpath)
for img in tqdm(os.listdir()):
    if img.endswith('jpg'):
        try:
            i=cv2.imread(img)
            img=img.replace('.jpg','.png')
            o=cv2.imwrite(newpath+img,i)
        except:
            print(img)


