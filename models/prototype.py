from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class S_Module(nn.Module):
    def __init__(
            self,
            inplanes=1024,
            planes=2048,
    ):
        super(S_Module, self).__init__()

        self.S1 = ConvModule(inplanes, planes, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)

    def forward(self, inputs):
        out = []
        feat3,feat4 = inputs[0],inputs[1]
        out.append(self.S1(feat3))
        out.append(feat4)
        return out

class ConvModule(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            kernel_size,
            stride,
            padding,
            bias=False,
            groups=32,
    ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.bn(x1)
        out = self.relu(x1)
        return out


class MidAggr(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            loss_weight=0.5
    ):
        super(MidAggr, self).__init__()
        self.convs = ConvModule(inplanes, inplanes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)

    def forward(self, x):
        x = self.convs(x)
        return x


class T_Module(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 downsample_scale=3,
                 ):
        super(T_Module, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False, groups=32)
        self.pool = nn.MaxPool3d((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Upsampling(nn.Module):
    def __init__(self,
                 scale=(2, 1, 1),
                 ):
        super(Upsampling, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return x


class Downampling(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=(3, 1, 1),
                 stride=(1, 1, 1),
                 padding=(1, 0, 0),
                 bias=False,
                 groups=32,
                 norm=False,
                 activation=False,
                 downsample_position='after',
                 downsample_scale=(1, 2, 2),
                 ):

        super(Downampling, self).__init__()
        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        self.norm = nn.BatchNorm3d(planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if activation else None
        assert (downsample_position in ['before', 'after'])
        self.downsample_position = downsample_position
        self.pool = nn.MaxPool3d(downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        if self.downsample_position == 'before':
            x = self.pool(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.downsample_position == 'after':
            x = self.pool(x)

        return x


class CrossFusion(nn.Module):
    def __init__(self,
                 in_channels=[1024, 1024],
                 mid_channels=[1024, 1024],
                 out_channels=2048
                 ):
        super(CrossFusion, self).__init__()

        self.m1 = ConvModule(in_channels[0],mid_channels[0],kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False, groups=32)
        self.m2 = ConvModule(in_channels[1],mid_channels[1],kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False, groups=32)

        in_dims = np.sum(mid_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_dims, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        out = [self.m1(inputs[0]),self.m2(inputs[1])]
        out = torch.cat(out, 1)
        out = self.fusion_conv(out)
        return out

class Prototype(nn.Module):

    def __init__(self,
                 in_channels=[1024, 2048],
                 out_channels=1024
                 ):
        super(Prototype, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.t_modulation_ops = nn.ModuleList()
        self.upsampling_ops = nn.ModuleList()
        self.level_fusion_op = CrossFusion()
        self.s_modulation = S_Module()

        for i in range(0, self.num_ins, 1):
            t_modulation = T_Module(2048,1024)
            self.t_modulation_ops.append(t_modulation)

        self.downsampling = Downampling(1024, 1024, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                                       padding=(1, 0, 0), bias=False,downsample_scale = (1, 1, 1))

        out_dims =2048
        self.level_fusion_op2 = CrossFusion()

        self.pyramid_fusion_op = nn.Sequential(
            nn.Conv3d(out_dims * 2, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True)
        )
        self.mid_aggr_3 = MidAggr(1024,1024)
        self.mid_aggr_2 = MidAggr(256,256)
        self.mid_aggr_1 = MidAggr(128, 128)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs_all):

        feat4,feat3,feat2,feat1= inputs_all[0],inputs_all[1],inputs_all[2],inputs_all[3]

        feat3_new = self.mid_aggr_3(feat3)
        feat2_new = self.mid_aggr_2(feat2)
        feat1_new = self.mid_aggr_1(feat1)

        inputs = [feat3,feat4]
        outs = self.s_modulation(inputs)

        outs = [t_modulation(outs[i]) for i, t_modulation in enumerate(self.t_modulation_ops)]
        outs0 = outs

        outs[1] = outs[1]+ outs[0]
        outs1 = self.level_fusion_op2(outs)
        outs = outs0

        if self.downsampling is not None:
            outs[1] = outs[1]+self.downsampling(outs[0])
        outs = self.level_fusion_op(outs)

        outs = self.pyramid_fusion_op(torch.cat([outs1, outs], 1))
        outs = outs.squeeze(2)
        feat3_new =feat3_new.squeeze(2)
        feat2_new = feat2_new.squeeze(2)
        feat1_new = feat1_new.squeeze(2)
        return outs,feat3_new,feat2_new,feat1_new