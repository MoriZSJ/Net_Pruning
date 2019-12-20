# full assembly of the sub-parts to form the complete net

'''
####Unet模型
import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)

    
'''


##### use UNET++ model
import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
   def __init__(self, n_channels, n_classes):
       super(UNet, self).__init__()
       self.inc = inconv(n_channels, 64)              
       self.down1 = down(64, 128)                     
       self.up0_1 = up(192, 96)
       self.down2 = down(128, 256)
       self.up1_1 = up(384, 192)
       self.up0_2 = up2(352, 176)
       self.down3 = down(256, 512)
       self.up2_1 = up(768, 384)
       self.up1_2 = up2(704, 352)
       self.up0_3 = up2(592, 296)
       self.down4 = down(512, 512)
       self.up1 = up(1024, 256)
       self.up2 = up2(896, 128)
       self.up3 = up2(608, 64)
       self.up4 = up2(424, 64)
       self.outc = outconv(64, n_classes)

   def forward(self, x):
       x0_0 = self.inc(x)
       x1_0 = self.down1(x0_0)
       x0_1 = self.up0_1(x1_0, x0_0)
       x2_0 = self.down2(x1_0)
       x1_1 = self.up1_1(x2_0, x1_0)
       x0_2 = self.up0_2(x1_1, x0_1, x0_0)
       x3_0 = self.down3(x2_0)
       x2_1 = self.up2_1(x3_0, x2_0)
       x1_2 = self.up1_2(x2_1, x1_1, x1_0)
       x0_3 = self.up0_3(x1_2, x0_2, x0_0)
       x4_0 = self.down4(x3_0)
       x = self.up1(x4_0, x3_0)
       x = self.up2(x, x2_1, x2_0)
       x = self.up3(x, x1_2, x1_0)
       x = self.up4(x, x0_3, x0_0)
       x = self.outc(x)
       return F.sigmoid(x)




###### official Unet++ model
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from .unet_parts import concat

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print("x:",x.size())
        out = self.conv1(x)
        # print(out.size())
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


class NestedUNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(args.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.concat = concat()

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print(self.up(x1_0).size()[2])
        x0_1 = self.conv0_1(self.concat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        # print(x1_0.size(),x2_0.size())    # [1, 64, 307, 307],[1, 128, 153, 153]
        x1_1 = self.conv1_1(self.concat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(self.concat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.concat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(self.concat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(self.concat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.concat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(self.concat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(self.concat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(self.concat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)

            return (F.sigmoid(output1), F.sigmoid(output2), F.sigmoid(output3), F.sigmoid(output4))

        else:
            output = self.final(x0_4)
            return F.sigmoid(output)






'''
if __name__ == "__main__":
    img = torch.rand(1,3,256,256)
    gt = torch.rand(1,1,256,256)
    model = UNet(n_channels=3, n_classes=1)
    result = model(img)
    criterion = torch.nn.BCELoss() 
    loss = criterion(result.squeeze(), gt.squeeze())
    loss.backward()
    print(model.up0_2.conv.conv[0].weight.grad)
    print(loss)
'''