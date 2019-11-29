import cv2
import numpy as np
import os
import matplotlib.pylab  as plt
import matplotlib.image as mpimg
LabelPath = 'F:\\S_Dataset\\new0522img\\'
outpath = 'F:\\S_Dataset\\new0522img3channels\\'
file = os.listdir(LabelPath)
    #LabelPath = 'D:\\testDataset\\neg\\label\\'
#outpath = 'D:\\testDataset\\neg\\graylabel\\'
#file = os.listdir('D:\\testDataset\\neg\\label\\')

for temp in file:

    img = cv2.imread(os.path.join(LabelPath,temp) ,0 )
    
    thresh3 = cv2.cvtColor( img, cv2.COLOR_GRAY2RGB )


    # ret,thresh3 = cv2.threshold(img,20,255,cv2.THRESH_BINARY)
    # thresh3 = rgb2gray(img)

    print(os.path.join(outpath, temp) )
    # img = cv2.imread(os.path.join(LabelPath,temp) ,0 )
    
    cv2.imwrite(os.path.join(outpath, temp), thresh3)