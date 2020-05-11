import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import Config
from nets.ssd_layers import Detect,L2Norm,PriorBox
from nets.vgg import vgg as add_vgg


class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = Config
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # 获得conv4_3的内容 relu层也算 Pooling不进行relu 一共36层
        for k in range(23):# 22层
            x = self.vgg[k](x)

        s = self.L2Norm(x)# L2标准化 原因：深度不够
        sources.append(s)

        # 获得fc7的内容
        for k in range(23, len(self.vgg)):#23-35
            x = self.vgg[k](x)
        sources.append(x)

        # 获得后面的内容
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

 # [batch_size,channel
        # 添加回归层和分类层
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())  # permute 通道数翻转
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 进行resize
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect.apply(loc.view(loc.size(0), -1, 4),
                                        self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                                        self.priors)
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
                    )
        return output


def add_extras(i, batch_norm=False):
    # 向VGG添加了额外的图层以进行特征缩放
    layers = []
    in_channels = i

    # Block 6
    # 19,19,1024 -> 10,10,512
    layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]  # [0]
    # C6_2
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

    # Block 7
    # 10,10,512 -> 5,5,256
    layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    # Block 8
    # 5,5,256 -> 3,3,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    # Block 9
    # 3,3,256 -> 1,1,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return layers


mbox = [4, 6, 6, 6, 4, 4]


def get_ssd(phase, num_classes):
    vgg, extra_layers = add_vgg(3), add_extras(1024)

    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):  # k是索引，v是枚举出来得对象(0 21) (1 -2)提取出21层和33层对应（4，6）
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 mbox[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  mbox[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):  # （2，1）（3，3）（4，5）（5，7）
        loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                  * num_classes, kernel_size=3, padding=1)]

    SSD_MODEL = SSD(phase, vgg, extra_layers, (loc_layers, conf_layers), num_classes)
    return SSD_MODEL
