import numpy as np
import cv2
from skimage import io, data
import random
import os

threshold = 100
imgDirPath = 'D:\\srcDataset\\newimg\\'
labelDirPath = 'D:\\srcDataset\\newlabel\\'
outImagePath = 'D:\\Dataset\\newneg\\img\\'
outLabelPath = 'D:\\Dataset\\newneg\\label\\'
patchPerImg = 4

def readImage(path, labelPath):
    img = cv2.imread(os.path.join(imgDirPath, path))
    labelImg = cv2.imread(os.path.join(labelDirPath, labelPath))
    return img, labelImg

def randomCrop(img, labelImg):
    while True:
        random_x = random.randint(0, 2064-259)
        random_y = random.randint(0, 1544-259)
        patch = img[random_y:random_y+256, random_x:random_x+256, :]
        labelpatch = labelImg[random_y:random_y+256, random_x:random_x+256, :]
        if np.sum(labelpatch[:,:,2] > 0) > 0:
            continue
        else:
            if np.sum(patch[:,:] > 50) < 10:
                continue
            else:
                return patch, labelpatch

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
            patch_img, patch_label = randomCrop(img, label)
            patch_img_name = imgPath.split('.')[0]+str(t)+'_neg.jpg'
            patch_label_name = labelPath.split('.')[0] + str(t) + '_neg.jpg'
            writeImage(patch_img, patch_label, patch_img_name, patch_label_name)
        
main()