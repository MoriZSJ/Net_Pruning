import sys
import os
import pdb
import cv2
import copy

def readroi(f):
    with open(f) as of:
        ser = of.read().split("\n")
        n = 1
	ch = ser[0].split(',')
        num = ch[0]
        for j in range(len(ser)-2):
            per = ser[n].rfind(",")
            per = ser[n].rfind(",", 0, per-1)
            ser[n] = ser[n][:per].replace(",", " ")
            n = n+1
        ser.pop()
        del ser[0]
        return ser,num

def listFile(dr, filter):
    fs = os.listdir(dr)
    for i in range(len(fs)-1, -1, -1):
        if not fs[i].lower().endswith(filter):
            del fs[i]
    return fs

root = "WIDER_train/images"
fs = listFile(root, ".xml.txt")
#pdb.set_trace()
with open("wider_face_train_bbx_gt.txt", "w") as of:
    for i in range(len(fs)):
        png =fs[i][:-8] + ".png"
        png_h =fs[i][:-8] + "_h.png"
        png_v =fs[i][:-8] + "_v.png"
        png_hv =fs[i][:-8] + "_hv.png"
        # image = cv2.imread(png)
        # image_h = cv2.flip(image,1)#horizontal
        # image_v = cv2.flip(image,0)#vertical
        # image_hv = cv2.flip(image,-1)#both
        # cv2.imwrite(png_h, image_h)
        # cv2.imwrite(png_v, image_v)
        # cv2.imwrite(png_hv, image_hv)
        #pdb.set_trace()
        roi,num = readroi(root + "/" + fs[i][:-8] + ".xml.txt")
        # line = png + " "
        # line_h = png_h + " "
        # line_v = png_v + " " 
        # line_hv = png_hv + " "
	if num == '0':
		continue
        png = png + "\n" + num
        png_h = png_h +  "\n" + num
        png_v = png_v + "\n" + num
        png_hv = png_hv +  "\n" + num
        for k in roi:

            tmp = k.split(' ')
            tmp_h=copy.deepcopy(tmp)
            tmp_v=copy.deepcopy(tmp)
            tmp_hv=copy.deepcopy(tmp)
            tmp_h[0] = str(2560 - int(tmp[2]))
            tmp_h[2] = str(2560 - int(tmp[0]))
            tmp_v[1] = str(1920 - int(tmp[3]))
            tmp_v[3] = str(1920 - int(tmp[1]))
            tmp_hv[0] = str(2560 - int(tmp[2]))
            tmp_hv[1] = str(1920 - int(tmp[3]))
            tmp_hv[2] = str(2560 - int(tmp[0]))
            tmp_hv[3] = str(1920 - int(tmp[1]))
            tmp[2] = str(int(tmp[2]) - int(tmp[0]))
            tmp[3] = str(int(tmp[3]) - int(tmp[1]))
            tmp_h[2] = str(int(tmp_h[2]) - int(tmp_h[0]))
            tmp_h[3] = str(int(tmp_h[3]) - int(tmp_h[1]))
            tmp_v[2] = str(int(tmp_v[2]) - int(tmp_v[0]))
            tmp_v[3] = str(int(tmp_v[3]) - int(tmp_v[1]))
            tmp_hv[2] = str(int(tmp_hv[2]) - int(tmp_hv[0]))
            tmp_hv[3] = str(int(tmp_hv[3]) - int(tmp_hv[1]))
            k = tmp[0]+' '+tmp[1]+' '+tmp[2]+' '+tmp[3]+" 0 0 0 0 0 "
            kh = tmp_h[0]+' '+tmp_h[1]+' '+tmp_h[2]+' '+tmp_h[3]+" 0 0 0 0 0 "
            kv = tmp_v[0]+' '+tmp_v[1]+' '+tmp_v[2]+' '+tmp_v[3]+" 0 0 0 0 0 "
            khv = tmp_hv[0]+' '+tmp_hv[1]+' '+tmp_hv[2]+' '+tmp_hv[3]+" 0 0 0 0 0 "
            #pdb.set_trace()
            png = png + "\n" + k
            png_h = png_h + "\n" + kh 
            png_v = png_v + "\n" + kv 
            png_hv = png_hv + "\n" + khv

        png = png + "\n"
        of.write(png)
        png_h = png_h + "\n"
        of.write(png_h)
        png_v = png_v + "\n"
        of.write(png_v)
        png_hv = png_hv + "\n"
        of.write(png_hv)

