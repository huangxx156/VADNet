from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        """
        :param indim:    512
        :param outdim:   512
        :param stride:   1
        """
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        """
        :param inplanes:
        :param planes:
        :param scale_factor:
        """
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        """
        :param f:
        :param pm:
        :return:
        """
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim):
        """
        :param mdim:
        """
        super(Decoder, self).__init__()
        self.convFM4 = nn.Conv2d(2048+1024, 1024, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM4 = ResBlock(1024, mdim)

        self.convFM3 = nn.Conv2d(256+256, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM3 = ResBlock(mdim, 128)

        self.convFM2 = nn.Conv2d(128+128, 128, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM2 = ResBlock(128, 128)

        self.deconv1 = nn.Conv2d(512,256, kernel_size=(3, 3), padding=(1, 1), stride=1)


        self.pred2 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=(3, 3), padding=(1, 1), stride=1),
            nn.ReLU(inplace=True))

    def forward(self,  feat_mid, feat_mid3, feat_mid2,feat_mid1):
        feat_mid = torch.cat([feat_mid,feat_mid3],dim=1)
        decon4 = self.ResMM4(self.convFM4(feat_mid))
        decon4 = F.interpolate(decon4, scale_factor=2, mode='bilinear', align_corners=False)

        feat_mid = torch.cat([decon4,feat_mid2],dim=1)
        decon3 = self.ResMM3(self.convFM3(feat_mid))
        decon3 = F.interpolate(decon3, scale_factor=2, mode='bilinear', align_corners=False)

        feat_mid = torch.cat([decon3, feat_mid1], dim=1)
        decon2 = self.ResMM2(self.convFM2(feat_mid))
        decon2 = F.interpolate(decon2, scale_factor=2, mode='bilinear', align_corners=False)

        p2 = self.pred2(F.relu(decon2))

        return p2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def Norm2d(in_channels):
    normalizationLayer = torch.nn.BatchNorm2d(in_channels)
    return normalizationLayer

def bnrelu(channels):
    return nn.Sequential(Norm2d(channels),
                         nn.ReLU(inplace=True))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class IdentityResidualBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=bnrelu,
                 dropout=None,
                 dist_bn=False
                 ):
        super(IdentityResidualBlock, self).__init__()
        self.dist_bn = dist_bn

        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2d(in_channels,
                                    channels[0],
                                    3,
                                    stride=stride,
                                    padding=dilation,
                                    bias=False,
                                    dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1],
                                    3,
                                    stride=1,
                                    padding=dilation,
                                    bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1",
                 nn.Conv2d(in_channels,
                           channels[0],
                           1,
                           stride=stride,
                           padding=0,
                           bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0],
                                    channels[1],
                                    3, stride=1,
                                    padding=dilation, bias=False,
                                    groups=groups,
                                    dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2],
                                    1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)
        return out

class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)

class wider_resnet38_a2(nn.Module):

    def __init__(self,
                 structure,
                 norm_act=bnrelu,
                 classes=0,
                 dilation=False,
                 dist_bn=False
                 ):
        super(wider_resnet38_a2, self).__init__()
        self.dist_bn = dist_bn
        nn.Dropout = nn.Dropout2d
        norm_act = bnrelu
        self.structure = structure
        self.dilation = dilation

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        self.mod1 = torch.nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        ]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                    (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1

                if mod_id == 4:
                    drop = partial(nn.Dropout, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout, p=0.5)
                else:
                    drop = None

                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels,
                                          channels[mod_id], norm_act=norm_act,
                                          stride=stride, dilation=dil,
                                          dropout=drop, dist_bn=self.dist_bn)
                ))

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id < 2:
                self.add_module("pool%d" %
                                (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, "classifier"):
            return self.classifier(out)
        else:
            return out


class Encoder(nn.Module):
    def __init__(self, num_classes = 19):
        """
        :param num_classes:
        :param trunk:
        :param criterion:
        """

        super(Encoder, self).__init__()
        # self.criterion = criterion
        self.num_classes = num_classes

        wide_resnet = wider_resnet38_a2(structure =[3, 3, 6, 3, 1, 1], classes=1000, dilation=True)
        wide_resnet = torch.nn.DataParallel(wide_resnet)

        wide_resnet = wide_resnet.module
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        self.interpolate = F.interpolate
        del wide_resnet

    def forward(self, imgs):
        feature5, feature4, feature3,feature2 = [], [], [], []
        for t in range(len(imgs)):
            image = imgs[t]
            # res 1
            m1 = self.mod1(image)

            # res 2
            m2 = self.mod2(self.pool2(m1))

            # res 3
            m3 = self.mod3(self.pool3(m2))

            # res 4-7
            m4 = self.mod4(m3)
            m5 = self.mod5(m4)
            m6 = self.mod6(m5)

            feature2.append(m2.unsqueeze(1))
            feature3.append(m3.unsqueeze(1))
            feature4.append(m5.unsqueeze(1))
            feature5.append(m6.unsqueeze(1))

        feat_matrix5 = torch.cat(feature5, dim=1)
        feat_matrix4 = torch.cat(feature4,dim=1)
        feat_matrix3 = torch.cat(feature3, dim=1)
        feat_matrix2 = torch.cat(feature2, dim=1)

        feat_matrix5 = feat_matrix5.permute(0,2,1,3,4)
        feat_matrix4 = feat_matrix4.permute(0, 2, 1, 3, 4)
        feat_matrix3 = feat_matrix3.permute(0, 2, 1, 3, 4)
        feat_matrix2 = feat_matrix2.permute(0, 2, 1, 3, 4)

        return feat_matrix5,feat_matrix4,feat_matrix3,feat_matrix2
