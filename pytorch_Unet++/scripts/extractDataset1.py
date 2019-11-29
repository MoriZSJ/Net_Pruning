import numpy as np
import cv2
from skimage import io, data
import random
import os
import time 
threshold = 40
imgDirPath = 'F:\\S_SrcDataset\\newimg0522\\'
labelDirPath = 'F:\\S_SrcDataset\\newlabel0522\\'
outImagePath = 'F:\\S_Dataset\\new0522img\\'
outLabelPath = 'F:\\S_Dataset\\new0522label\\'
patchPerImg = 20

def readImage(path, labelPath):
    img = cv2.imread(os.path.join(imgDirPath, path))
    labelImg = cv2.imread(os.path.join(labelDirPath, labelPath))
    return img, labelImg

def randomCrop(img, labelImg):
    start3 = time.clock()
    while True:
        random_x = random.randint(0, img.shape[1]-259)
        random_y = random.randint(0, img.shape[0]-259)
        patch = img[random_y:random_y+256, random_x:random_x+256, :]
        labelpatch = labelImg[random_y:random_y+256, random_x:random_x+256, :]
        end3 = time.clock()
        flag=1
        # print('time:%s Seconds'%(end3-start3))
        if (end3-start3)>2:
            flag=0
            return flag,patch, labelpatch
        if np.sum(labelpatch[:,:,2] > 0) < threshold:
            continue

        else:
            return flag,patch, labelpatch

def writeImage(img, label, imgName, labelName):
    cv2.imwrite(os.path.join(outImagePath, imgName), img)
    cv2.imwrite(os.path.join(outLabelPath, labelName), label)


def main():
    imgFiles = sorted(os.listdir(imgDirPath))
    labelFiles = sorted(os.listdir(labelDirPath))
    for imgPath, labelPath in zip(imgFiles, labelFiles):
        img, label = readImage(imgPath, labelPath)
        
        for t in range(patchPerImg):
            print('processing ...:', imgPath)
            flag,patch_img, patch_label = randomCrop(img, label)
            if flag==0:
                continue
            patch_img_name = imgPath.split('.')[0]+str(t)+'_add.bmp'
            patch_label_name = labelPath.split('.')[0] + str(t) + '_add.bmp'
            writeImage(patch_img, patch_label, patch_img_name, patch_label_name)

               
        
main()