# -!- coding: utf-8 -!-
import argparse
import copy
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

from models.c2former import C2Former, Indexer
# from models.ctf_fusion import Add2, GPT, Add
from models.fusion import TransformerFusionBlock

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
from utils.tal.anchor_generator import make_anchors, dist2bbox
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

class DDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max(
            (ch[0], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in
            ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(3):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x[:3]], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)

class DualDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.num_head = len(ch) // 3
        self.nc = nc  # number of classes
        self.nl = len(ch) // self.num_head  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        if self.num_head == 3:
            c6, c7 = make_divisible(max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), 4), max((ch[self.nl * 2], min((self.nc * 2, 128))))  # channels
        else:
            if self.num_head == 5:
                c6, c7 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
                c8, c9 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
                c10, c11 = make_divisible(max((ch[self.nl * 4] // 4, self.reg_max * 4, 16)), 4), max((ch[self.nl * 4], min((self.nc * 4, 128))))  # channels
            elif self.num_head == 7:
                c6, c7 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
                c8, c9 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
                c10, c11 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
                c12, c13 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
                c14, c15 = make_divisible(max((ch[self.nl * 6] // 4, self.reg_max * 4, 16)), 4), max((ch[self.nl * 6], min((self.nc * 6, 128))))  # channels

        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl*1])
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl*1])
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:self.nl*2])
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:self.nl*2])

        if self.num_head == 3:
            self.cv6 = nn.ModuleList(nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3, g=4), nn.Conv2d(c6, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*2:self.nl*3])
            self.cv7 = nn.ModuleList(nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
        else:
            if self.num_head == 5:
                self.cv6 = nn.ModuleList(nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3, g=4), nn.Conv2d(c6, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*2:self.nl*3])
                self.cv7 = nn.ModuleList(nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
                self.cv8 = nn.ModuleList(nn.Sequential(Conv(x, c8, 3), Conv(c8, c8, 3, g=4), nn.Conv2d(c8, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*3:self.nl*4])
                self.cv9 = nn.ModuleList(nn.Sequential(Conv(x, c9, 3), Conv(c9, c9, 3), nn.Conv2d(c9, self.nc, 1)) for x in ch[self.nl*3:self.nl*4])
                self.cv10 = nn.ModuleList(nn.Sequential(Conv(x, c10, 3), Conv(c10, c10, 3, g=4), nn.Conv2d(c10, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*4:self.nl*5])
                self.cv11 = nn.ModuleList(nn.Sequential(Conv(x, c11, 3), Conv(c11, c11, 3), nn.Conv2d(c11, self.nc, 1)) for x in ch[self.nl*4:self.nl*5])
            elif self.num_head == 7:
                self.cv6 = nn.ModuleList(nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3, g=4), nn.Conv2d(c6, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*2:self.nl*3])
                self.cv7 = nn.ModuleList(nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
                self.cv8 = nn.ModuleList(nn.Sequential(Conv(x, c8, 3), Conv(c8, c8, 3, g=4), nn.Conv2d(c8, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*3:self.nl*4])
                self.cv9 = nn.ModuleList(nn.Sequential(Conv(x, c9, 3), Conv(c9, c9, 3), nn.Conv2d(c9, self.nc, 1)) for x in ch[self.nl*3:self.nl*4])

                self.cv10 = nn.ModuleList(nn.Sequential(Conv(x, c10, 3), Conv(c10, c10, 3, g=4), nn.Conv2d(c10, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*4:self.nl*5])
                self.cv11 = nn.ModuleList(nn.Sequential(Conv(x, c11, 3), Conv(c11, c11, 3), nn.Conv2d(c11, self.nc, 1)) for x in ch[self.nl*4:self.nl*5])
                self.cv12 = nn.ModuleList(nn.Sequential(Conv(x, c12, 3), Conv(c12, c12, 3, g=4), nn.Conv2d(c12, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*5:self.nl*6])
                self.cv13 = nn.ModuleList(nn.Sequential(Conv(x, c13, 3), Conv(c13, c13, 3), nn.Conv2d(c13, self.nc, 1)) for x in ch[self.nl*5:self.nl*6])
                self.cv14 = nn.ModuleList(nn.Sequential(Conv(x, c14, 3), Conv(c14, c14, 3, g=4), nn.Conv2d(c14, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*6:])
                self.cv15 = nn.ModuleList(nn.Sequential(Conv(x, c15, 3), Conv(c15, c15, 3), nn.Conv2d(c15, self.nc, 1)) for x in ch[self.nl*6:])

        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)
        if self.num_head == 3:
            self.dfl3 = DFL(self.reg_max)
        else:
            if self.num_head == 5:
                self.dfl3 = DFL(self.reg_max)
                self.dfl4 = DFL(self.reg_max)
                self.dfl5 = DFL(self.reg_max)
            elif self.num_head == 7:
                self.dfl3 = DFL(self.reg_max)
                self.dfl4 = DFL(self.reg_max)
                self.dfl5 = DFL(self.reg_max)
                self.dfl6 = DFL(self.reg_max)
                self.dfl7 = DFL(self.reg_max)

    def forward(self, x):
        num_head = len(x) // 3
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        if num_head == 3:
            d3 = []
        else:
            if num_head == 5:
                d3 = []
                d4 = []
                d5 = []
            elif num_head == 7:
                d3 = []
                d4 = []
                d5 = []
                d6 = []
                d7 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl + i]), self.cv5[i](x[self.nl + i])), 1))
            if num_head == 3:
                d3.append(torch.cat((self.cv6[i](x[self.nl * 2 + i]), self.cv7[i](x[self.nl * 2 + i])), 1))
            else:
                if num_head == 5:
                    d3.append(torch.cat((self.cv6[i](x[self.nl * 2 + i]), self.cv7[i](x[self.nl * 2 + i])), 1))
                    d4.append(torch.cat((self.cv8[i](x[self.nl * 3 + i]), self.cv9[i](x[self.nl * 3 + i])), 1))
                    d5.append(torch.cat((self.cv10[i](x[self.nl * 4 + i]), self.cv11[i](x[self.nl * 4 + i])), 1))
                elif num_head == 7:
                    d3.append(torch.cat((self.cv6[i](x[self.nl * 2 + i]), self.cv7[i](x[self.nl * 2 + i])), 1))
                    d4.append(torch.cat((self.cv8[i](x[self.nl * 3 + i]), self.cv9[i](x[self.nl * 3 + i])), 1))
                    d5.append(torch.cat((self.cv10[i](x[self.nl * 4 + i]), self.cv11[i](x[self.nl * 4 + i])), 1))
                    d6.append(torch.cat((self.cv12[i](x[self.nl * 5 + i]), self.cv13[i](x[self.nl * 5 + i])), 1))
                    d7.append(torch.cat((self.cv14[i](x[self.nl * 6 + i]), self.cv15[i](x[self.nl * 6 + i])), 1))
        if self.training:
            if num_head == 3:
                return [d1, d2, d3]
            else:
                if num_head == 5:
                    return [d1, d2, d3, d4, d5]
                elif num_head == 7:
                    return [d1, d2, d3, d4, d5, d6, d7]
                else:
                    return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box1, cls1 = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox1 = dist2bbox(self.dfl(box1), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if num_head == 3:
            box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
            dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        else:
            if num_head == 5:

                box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
                dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

                box4, cls4 = torch.cat([di.view(shape[0], self.no, -1) for di in d4], 2).split((self.reg_max * 4, self.nc), 1)
                dbox4 = dist2bbox(self.dfl4(box4), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

                box5, cls5 = torch.cat([di.view(shape[0], self.no, -1) for di in d5], 2).split((self.reg_max * 4, self.nc), 1)
                dbox5 = dist2bbox(self.dfl5(box5), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

            elif num_head == 7:

                box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
                dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

                box4, cls4 = torch.cat([di.view(shape[0], self.no, -1) for di in d4], 2).split((self.reg_max * 4, self.nc), 1)
                dbox4 = dist2bbox(self.dfl4(box4), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

                box5, cls5 = torch.cat([di.view(shape[0], self.no, -1) for di in d5], 2).split((self.reg_max * 4, self.nc), 1)
                dbox5 = dist2bbox(self.dfl5(box5), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

                box6, cls6 = torch.cat([di.view(shape[0], self.no, -1) for di in d6], 2).split((self.reg_max * 4, self.nc), 1)
                dbox6 = dist2bbox(self.dfl6(box6), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

                box7, cls7 = torch.cat([di.view(shape[0], self.no, -1) for di in d7], 2).split((self.reg_max * 4, self.nc), 1)
                dbox7 = dist2bbox(self.dfl7(box7), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if num_head == 3:
                y = [torch.cat((dbox1, cls1.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1)]
        else:
            if num_head == 5:
                y = [torch.cat((dbox1, cls1.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1), torch.cat((dbox4, cls4.sigmoid()), 1), torch.cat((dbox5, cls5.sigmoid()), 1)]

            elif num_head == 7:
                y = [torch.cat((dbox1, cls1.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1), torch.cat((dbox4, cls4.sigmoid()), 1), torch.cat((dbox5, cls5.sigmoid()), 1), torch.cat((dbox6, cls6.sigmoid()), 1), torch.cat((dbox7, cls7.sigmoid()), 1)]

            else:
                y = [torch.cat((dbox1, cls1.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]

        if num_head == 3:
            return y if self.export else (y, [d1, d2, d3])
        else:
            if num_head == 5:
                return y if self.export else (y, [d1, d2, d3, d4, d5])
            elif num_head == 7:
                return y if self.export else (y, [d1, d2, d3, d4, d5, d6, d7])
            else:
                return y if self.export else (y, [d1, d2])
        
        # y = torch.cat((dbox2, cls2.sigmoid()), 1)
        # return y if self.export else (y, d2)
        # y1 = torch.cat((dbox, cls.sigmoid()), 1)
        # y2 = torch.cat((dbox2, cls2.sigmoid()), 1)
        # return [y1, y2] if self.export else [(y1, d1), (y2, d2)]
        # return [y1, y2] if self.export else [(y1, y2), (d1, d2)]

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)

        if self.num_head == 3:
            for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        else:
            if self.num_head == 5:
                for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
                    a[-1].bias.data[:] = 1.0  # box
                    b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
                for a, b, s in zip(m.cv8, m.cv9, m.stride):  # from
                    a[-1].bias.data[:] = 1.0  # box
                    b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
                for a, b, s in zip(m.cv10, m.cv11, m.stride):  # from
                    a[-1].bias.data[:] = 1.0  # box
                    b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
            elif self.num_head == 7:
                for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
                    a[-1].bias.data[:] = 1.0  # box
                    b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
                for a, b, s in zip(m.cv8, m.cv9, m.stride):  # from
                    a[-1].bias.data[:] = 1.0  # box
                    b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
                for a, b, s in zip(m.cv10, m.cv11, m.stride):  # from
                    a[-1].bias.data[:] = 1.0  # box
                    b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
                for a, b, s in zip(m.cv12, m.cv13, m.stride):  # from
                    a[-1].bias.data[:] = 1.0  # box
                    b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
                for a, b, s in zip(m.cv14, m.cv15, m.stride):  # from
                    a[-1].bias.data[:] = 1.0  # box
                    b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)

class BaseModel(nn.Module):
    # YOLO base model
    def forward(self, x, x2, profile=False, visualize=False):
        return self._forward_once(x, x2, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, x2, profile=False, visualize=False):
        y, dt = [], []  # outputs
        # self.save
        # [0, 5, 6, 7, 9, 15, 15, 17, 17, 19, 19, 20, 21, 23, 26, 29, 31, 35, 36, 37, 37, 38, 38, 38, 44, 47, 50]

        for m in self.model:

            if m.f != -1:  # if not from previous layer
                if m.f != -4:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)

            # x = m(x)  # run

            if m.f == -4:
                x = m(x2)
            else:
                x = m(x)

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (RepConvN)) and hasattr(m, 'fuse_convs'):
                m.fuse_convs()
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (
        DDetect, DualDDetect)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
            # m.grid = list(map(fn, m.grid))
        return self


class DetectionModel(BaseModel):
    # YOLO detection model
    def __init__(self, cfg='yolo.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (DDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, DSegment, Panoptic)) else self.forward(x)
            # m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            # check_anchor_order(m)
            # m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            m.bias_init()  # only run once
        if isinstance(m, (DualDDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # forward = lambda x: self.forward(x)[0][0] if isinstance(m, (DualDSegment)) else self.forward(x)[0]
            # m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            # check_anchor_order(m)
            # m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, x2, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x, x2)  # augmented inference, None
        return self._forward_once(x, x2, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x, x2):
        print(f'开启测试增强')
        img_size = x.shape[-2:]  # height, width
        print(x.shape, x2.shape)
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        model_out = None
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            xi2 = scale_img(x2.flip(fi) if fi else x2, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi, xi2)  # forward
            if si == 1:
                model_out = yi[1]
            yi = yi[0]
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            # print(yi.shape)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), model_out  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLO augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y


Model = DetectionModel  # retain YOLO 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLO segmentation model
    def __init__(self, cfg='yolo-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLO classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLO classification model from a YOLO detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLO classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        RepConvN.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv, AConv, ConvTranspose,
            Bottleneck, SPP, SPPF, DWConv, BottleneckCSP, nn.ConvTranspose2d, DWConvTranspose2d, SPPCSPC, ADown,
            RepNCSPELAN4, SPPELAN, RepNCSPELAN4SCConv}:
            if m is Conv and args[0] == 64:  # new
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]

            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in {BottleneckCSP, SPPCSPC}:
                    args.insert(2, n)  # number of repeats
                    n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        # TODO: channel, gw, gd
        elif m in {DDetect, DualDDetect}:
            args.append([ch[x] for x in f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is Silence:
            c2 = 3
        elif m is TransformerFusionBlock:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        # elif m is Add:
        #
        #     c2 = ch[f[0]]
        #     args = [c2]
        # elif m is Add2:
        #     c2 = ch[f[0]]
        #     args = [c2, args[1]]
        # elif m is GPT:
        #     c2 = ch[f[0]]
        #     args = [c2]
        elif m is C2Former:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        elif m is Indexer:
            c2 = ch[f]
            args = [c2, args[1]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolo.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)
    model.eval()

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()