import numpy as np
import cv2
from skimage import io, data
import random
import os
import time 
threshold = 30
ResultimgDirPath = 'F:\\S_SrcDataset\\img_S_0520\\'
ResultlabelDirPath = 'F:\\S_SrcDataset\\img_S_0525Result\\'
SourceimgDirPath = 'F:\\S_SrcDataset\\img_S_0520\\'
SourcelabelDirPath = 'F:\\S_SrcDataset\\label_S_0520\\'
outImagePath = 'F:\\S_Dataset\\neg\\negimg0521\\'
outLabelPath = 'F:\\S_Dataset\\neg\\neglabel0521\\'

patchPerImg = 12

def readImage(path, labelPath, SlabelPath):
    img = cv2.imread(os.path.join(ResultimgDirPath, path))
    labelImg = cv2.imread(os.path.join(ResultlabelDirPath, labelPath),0)
    ret,labelImg = cv2.threshold(labelImg,10,255,cv2.THRESH_BINARY)
    SlabelImg = cv2.imread(os.path.join(SourcelabelDirPath, SlabelPath),0)
    ret,SImg = cv2.threshold(SlabelImg,10,255,cv2.THRESH_BINARY)
    return img, labelImg, SImg

def randomCrop(img, labelImg, Slabel):
    start3 = time.clock()
    while True:
        random_x = random.randint(0, img.shape[1]-259)
        random_y = random.randint(0, img.shape[0]-259)
        patch = img[random_y:random_y+256, random_x:random_x+256, :]
        labelpatch = labelImg[random_y:random_y+256, random_x:random_x+256]
        Slabelpatch = Slabel[random_y:random_y+256, random_x:random_x+256]
        end3 = time.clock()
        
        # print('time:%s Seconds'%(end3-start3))
        if (end3-start3)>3:
            
            return patch, labelpatch, Slabelpatch
        if np.sum(labelpatch[:,:] > 0) < threshold:
            continue
        else:
            return patch, labelpatch, Slabelpatch

def writeImage(img, label, imgName, labelName):
    cv2.imwrite(os.path.join(outImagePath, imgName), img)
    cv2.imwrite(os.path.join(outLabelPath, labelName), label)



def main():
    er = 0
    RimgFiles = sorted(os.listdir(ResultimgDirPath))
    RlabelFiles = sorted(os.listdir(ResultlabelDirPath))
    #SimgFiles = sorted(os.listdir(SourceimgDirPath))
    SlabelFiles = sorted(os.listdir(SourcelabelDirPath))
    for RimgPath, RlabelPath,SlabelPath in zip(RimgFiles, RlabelFiles,SlabelFiles):
        Rimg, Rlabel, Slabel = readImage(RimgPath, RlabelPath, SlabelPath)
        for t in range(patchPerImg):
            print('processing ...:', RimgPath)
            patch_img, patch_label, patch_Slabel = randomCrop(Rimg, Rlabel, Slabel)
            # if flag==0:
            #     break
            source = np.sum(patch_Slabel[:,:] > 0)
            #print(source)
            # if(patch_label.shape!=patch_Slabel.shape):
            #     continue
            and_Img = cv2.bitwise_and(patch_label,patch_Slabel)
            numofand = np.sum(and_Img[:,:] > 0)
            or_Img = cv2.bitwise_or(patch_label, patch_Slabel)
            numofor = np.sum(or_Img[:,:] > 0)
            if source == 0:
                label_name = SlabelPath.split('.')[0] + str(t) + '_emneg.bmp'
                img_name = RimgPath.split('.')[0]+str(t)+'_emneg.bmp'
                writeImage(patch_img, patch_Slabel, img_name, label_name)
                er = er + 1
                continue
            point1 = numofand/source
            #print(point1)
            point2 = numofor/source
            #print(point2)
            if point2 > 1.6:
                label_name = SlabelPath.split('.')[0] + str(t) + '_emover.bmp'
                img_name = RimgPath.split('.')[0]+str(t)+'_emover.bmp'
                writeImage(patch_img, patch_Slabel, img_name, label_name)
                continue
            if point1 < 0.4:
                label_name = SlabelPath.split('.')[0] + str(t) + '_emneg.bmp'
                img_name = RimgPath.split('.')[0]+str(t)+'_emneg.bmp'
                writeImage(patch_img, patch_Slabel, img_name, label_name)
                continue
            if point2 < 1.15 and point1 > 0.85:
                label_name = RlabelPath.split('.')[0] + str(t) + '_empos.bmp'
                img_name = RimgPath.split('.')[0]+str(t)+'_empos.bmp'
                writeImage(patch_img, patch_label, img_name, label_name)
                continue
    print(er)
            

            

        
main()
