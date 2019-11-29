import os
import shutil
sourceDir = "D:\\项目图\\0770\\Unet_S_label\\label\\new\\png"
ImgDataDir = "F:\\S_SrcDataset\\img"
LabelDataDir = "F:\\S_SrcDataset\\label"
files = os.listdir(sourceDir)
for f in files:
	sourceF = os.path.join(sourceDir,f)
	bimg = os.path.join(sourceF,'img.bmp')
	blabel = os.path.join(sourceF,'label.png')
	newname_img = os.path.join(ImgDataDir,f)+'.bmp'
	newname_label = os.path.join(LabelDataDir,f)+'.bmp'
	shutil.copyfile(bimg,newname_img)
	shutil.copyfile(blabel,newname_label)
