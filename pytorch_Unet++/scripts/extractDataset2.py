import numpy as np
import cv2
from skimage import io, data
import random
import os
import math
import time 
threshold = 40
imgDirPath = 'F:\\S_SrcDataset\\newimg0522\\'
labelDirPath = 'F:\\S_SrcDataset\\newlabel0522\\'
outImagePath = 'F:\\S_Dataset\\new0522img\\'
outLabelPath = 'F:\\S_Dataset\\new0522label\\'
patchPerImg = 10

def readImage(path, labelPath):
    img = cv2.imread(os.path.join(imgDirPath, path))
    labelImg = cv2.imread(os.path.join(labelDirPath, labelPath))
    return img, labelImg

def randomCrop(img, labelImg,imgPath,labelPath):
    start3 = time.clock()
    # ret, binary =  cv2.threshold(labelImg, 10, 255, cv2.THRESH_BINARY_INV)
    ret, binary =  cv2.threshold(labelImg, 10, 255, cv2.THRESH_BINARY)
    t=1
    while True:
        for i in range(0,4):
            for j in range(0,math.floor(img.shape[1]/256)):
                if (j+1)>=img.shape[1]:
                    x=img.shape[1]-259
                    y=i*256
                else:
                    x=j*256
                    y=i*256
                patch = img[y:y+256, x:x+256, :]
                labelpatch = binary[y:y+256, x:x+256, :]
                patch_img_name = imgPath.split('.')[0]+str(t)+'.bmp'
                patch_label_name = labelPath.split('.')[0] + str(t) + '.bmp'
                writeImage(patch, labelpatch, patch_img_name, patch_label_name)
                t=t+1
        return patch, labelpatch

def writeImage(img, label, imgName, labelName):
    cv2.imwrite(os.path.join(outImagePath, imgName), img)
    cv2.imwrite(os.path.join(outLabelPath, labelName), label)


def main():
    imgFiles = sorted(os.listdir(imgDirPath))
    labelFiles = sorted(os.listdir(labelDirPath))
    for imgPath, labelPath in zip(imgFiles, labelFiles):
        img, label = readImage(imgPath, labelPath)
        patch_img, patch_label = randomCrop(img, label,imgPath,labelPath)



               
        
main()