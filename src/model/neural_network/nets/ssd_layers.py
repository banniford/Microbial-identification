from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
from math import sqrt as sqrt
import numpy as np
from utils.box_utils import decode, nms
from utils.config import Config

class Detect(Function):
    # 回归预测结果，分类预测结果，先验框
    @staticmethod
    def forward(self, loc_data, conf_data, prior_data):
        if Config['nms_thresh'] <= 0:
            raise ValueError('nms_threshold must be non negative.')
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()
        # 图片数量 预测一般一张
        num = loc_data.size(0)  # batch size 一张图片
        # 先验框数量 8732
        num_priors = prior_data.size(0)
        # 存放输出(1,类别，200)
        output = torch.zeros(num, Config['num_classes'], Config["top_k"], 5)

        # 分类预测结果转换（1，8732，种类）torch.transpose(input, dim0, dim1, out=None) → Tensor 返回输入矩阵input的转置。交换维度dim0和dim1。 输出张量与输入张量共享内存，所以改变其中一个会导致另外一个也被修改。
        conf_preds = conf_data.view(num, num_priors,Config['num_classes']).transpose(2, 1)
        # 对每一张图片进行处理
        for i in range(num):
            # 对先验框解码获得预测框
            decoded_boxes = decode(loc_data[i], prior_data, Config['variance'])
            # 取出某一图片所有先验框种类
            conf_scores = conf_preds[i].clone()

            for cl in range(1, Config['num_classes']):
                # 对每一类进行非极大抑制
                # gt(a,b) 相当于 a > b conf_thresh阈值0.01 返回（True,False）
                c_mask = conf_scores[cl].gt(Config["conf_thresh"])
                # 两组合并去除false对应数据数据
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # 进行非极大抑制
                ids, count = nms(boxes, scores, Config['nms_thresh'], Config["top_k"])
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        # 进行排序
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        # 取出top_K框返回
        flt[(rank < Config["top_k"]).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            x,y = np.meshgrid(np.arange(f),np.arange(f))# 输入的x，y，就是网格点的横纵坐标列向量（非矩阵） 输出的X，Y，就是坐标矩阵。
            x = x.reshape(-1)# array([ 0,  1,  2, ..., 35, 36, 37])转成1维共1444个
            y = y.reshape(-1)#-1自动计算维度array([ 0,  0,  0, ..., 37, 37, 37])
            for i, j in zip(y,x):
                # zip()组合y和x形成[(0,0)(0,1)···(37,35)(37,36)(37,37)]
                # print(x,y)
                # 300/8 = 37.5
                f_k = self.image_size / self.steps[k]
                # 计算网格的中心
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 求短边 小正方形
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 求长边 大正方形
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 获得长方形
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # 获得所有的先验框
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)
        return output#clip函数裁剪最小值和最大值，防止低于0，超出1

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
