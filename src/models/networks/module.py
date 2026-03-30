import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)


def Norm_layer(norm_cfg, inplanes):

    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes,affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None, weight_std=False):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out

class ResEncoder(nn.Module):

    arch_settings = {
        10: (ResBlock, (3, 3, 3)),
        7: (ResBlock, (2, 2, 2)),
        4: (ResBlock, (1, 1, 1))
    }

    def __init__(self,
                 depth,
                 in_channels=1,
                 norm_cfg='BN',
                 activation_cfg='ReLU',
                 weight_std=False):
        super(ResEncoder, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.conv1 = conv3x3x3(in_channels, 64, kernel_size=3, stride=(1, 1, 1), padding=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, 64)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.layer1 = self._make_layer(block, 128, layers[0], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layers = []

        #

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='ReLU', weight_std=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3x3(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False, weight_std=weight_std), Norm_layer(norm_cfg, planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample, weight_std=weight_std))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        out.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)

        return out


class CAL_Net(nn.Module):
    def __init__(self, backbone_mask, num_classes=2):
        super().__init__()
        #self.oi_encode = backbone_oi
        self.mask_encode = backbone_mask
        # self.zoom_encode = backbone_zoom
        #self.feature_oi = None
        in_dim = 1536
        self.conv1 = nn.Conv3d(in_dim,  512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(512)
        self.relu = nn.ReLU()

        # self.query_conv = nn.Conv3d(in_dim, in_dim // 8, kernel_size=1)
        # self.key_conv = nn.Conv3d(in_dim, in_dim // 8, kernel_size=1)

        # if backbone_oi.model_type == "ResNet":
        #     value_in_dim = 512
        # elif backbone_oi.model_type == "DenseNet":
        #     value_in_dim = 1024
        # elif backbone_oi.model_type == "EfficientNetBN":
        #     value_in_dim = 1280
        # else:
        #     raise NotImplementedError(backbone_oi.model_type)
        value_in_dim = 512
        self.value_conv = nn.Conv3d(value_in_dim, value_in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc = nn.Linear(in_features=value_in_dim, out_features=num_classes, bias=True)

        self.transconv = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)

    def forward(self, x, y, z):
        # encode
        feature_re = x
        feature_tf = self.transconv(y)
        # feature_zoom = self.zoom_encode(y)[-2]
        feature_mask = self.mask_encode(z)[-1]
        features = torch.cat([feature_re, feature_tf, feature_mask], dim=1)
        features = self.conv1(features)
        features = self.bn1(features)
        features = self.relu(features)
        # decode
        # batch_size, channels, depth, height, width  = feature_oi.size()
        # proj_query = self.query_conv(feature_mask).view(batch_size, -1, width * height * depth).permute(0, 2, 1)  # B X CX(N)
        # proj_key = self.key_conv(feature_zoom).view(batch_size, -1, width * height * depth)  # B X C x (*W*H)
        # energy = torch.bmm(proj_query, proj_key)  # transpose check
        # attention = self.softmax(energy)  # BX (N) X (N)
        # proj_value = self.value_conv(feature_oi).view(batch_size, -1, width * height * depth)  # B X C X N
        # att_x = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # att_x = att_x.view(batch_size, channels, depth, height, width)
        # att_x = self.gamma * att_x + feature_oi
        # print('gama\n', self.gamma)
        # att_x = att_x + feature_oi
        # self.finalconv = att_x.clone()
        # classification head
        res = self.avgpool(features)
        res = res.view(res.size(0), -1)
        res = self.fc(res)

        return res
