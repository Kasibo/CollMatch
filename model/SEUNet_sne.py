"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
# from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch.nn as nn
from einops import rearrange
from torch.utils import model_zoo
import torch
import torch.nn.functional as F
# from model.basic import _ConvBNReLU
# from basic import _ConvBNReLU

# from att import *

__all__ = ['SENet', 'se_resnext50_32x4d', 'se_resnext101_32x4d']

pretrained_settings = {
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=64, input_3x3=True, downsample_kernel_size=1,
                 downsample_padding=0, num_classes=1000, **kwargs):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],  # 3
            groups=groups,  # 32
            reduction=reduction,  # 16
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],  # 4
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],  # 6
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],  # 3
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        # self.avg_pool = nn.AvgPool2d(7, stride=1)
        # self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        # self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        # self.up2_conv_reduce = _ConvBNReLU(512, 64, 1, **kwargs)
        # self.up3_conv_reduce = _ConvBNReLU(1024, 256, 1, **kwargs)
        # self.up4_conv_reduce = _ConvBNReLU(2048, 512, 1, **kwargs)
        # self.up5_conv_reduce = _ConvBNReLU(2048, 1024, 1, **kwargs)

        self.num_classes = num_classes

        #         self.relu = nn.ReLU(inplace=True)
        #         self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn1 = nn.BatchNorm2d(128)
        #         self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn2 = nn.BatchNorm2d(128)
        #         self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn3 = nn.BatchNorm2d(128)
        #         self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn4 = nn.BatchNorm2d(128)
        #         self.deconv5 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        #         self.bn5 = nn.BatchNorm2d(128)
        #         self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)input_3x3

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.deconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

        #         self.up2p_conv_reduce = _ConvBNReLU(256, 128, 1, **kwargs)
        #         self.up3p_conv_reduce = _ConvBNReLU(512, 128, 1, **kwargs)
        #         self.up4p_conv_reduce = _ConvBNReLU(1024, 128, 1, **kwargs)
        #         self.up5p_conv_reduce = _ConvBNReLU(2048, 128, 1, **kwargs)

        # self.up2p_conv_reduce = _ConvBNReLU(256, 128, 1, **kwargs)
        # self.up3p_conv_reduce = _ConvBNReLU(512, 256, 1, **kwargs)
        # self.up4p_conv_reduce = _ConvBNReLU(1024, 512, 1, **kwargs)
        # self.up5p_conv_reduce = _ConvBNReLU(2048, 1024, 1, **kwargs)

        # self.seg_conv_out = nn.Conv2d(128, num_classes, 1)
        
        self.corr = Corr(num_classes)
        self.proj = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x1 = self.layer0(x)  # b 64 64 64
        x2 = self.layer1(x1)  # b 256 64 64
        x3 = self.layer2(x2)  # b 512 32 32
        x4 = self.layer3(x3)  # b 1024 16 16
        x5 = self.layer4(x4)  # b 2048 8 8

        return x1, x2, x3, x4, x5

    def decoder(self, x1, x2, x3, x4, x5):
        """
        :param x2: c = 256
        :param x3: c = 512
        :param x4: c = 1024
        :param x5: c = 2048
        :return:
        """
        up = F.interpolate(self.up5_conv_reduce(x5), scale_factor=2, mode='bilinear')
        up = F.interpolate(self.up4_conv_reduce(torch.cat([up, x4], dim=1)), scale_factor=2, mode='bilinear',
                           align_corners=True)
        up = F.interpolate(self.up3_conv_reduce(torch.cat([up, x3], dim=1)), scale_factor=2, mode='bilinear',
                           align_corners=True)
        up = F.interpolate(self.up2_conv_reduce(torch.cat([up, x2], dim=1)), scale_factor=2, mode='bilinear',
                           align_corners=True)
        seg_out = self.seg_conv_out(
            torch.cat([up, F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        # edge_out = self.edge_conv_out(torch.cat([up, F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        seg_out = F.interpolate(seg_out, scale_factor=2, mode='bilinear', align_corners=True)
        # edge_out = F.interpolate(edge_out, scale_factor=2, mode='bilinear', align_corners=True)
        return seg_out

    def decoder_ConvTranspose(self, x1, x2, x3, x4, x5):
        """

        :rtype: object
        """
        score = self.bn5(self.relu(self.deconv5(x5)))  # b 1024 16 16
        score = score + x4
        score = self.bn4(self.relu(self.deconv4(score)))  # b 512 32 32
        score = score + x3
        score = self.bn3(self.relu(self.deconv3(score)))  # b 256 64 64
        score = score + x2
        score = self.bn2(self.relu(self.deconv2(score)))  # b 128 128 128s
        # if refined_features != None:
        #     score = torch.cat((score, F.interpolate(refined_features[-1],score.shape[-2:], mode='bilinear')),dim=1)
        score = self.bn1(self.relu(self.deconv1(score)))  # 8 128 256 256

        seg_out = self.classifier(score)
        return seg_out, score

#     def feature_local_mask(self, x, masking_prob=0.5, is_training=True, scale_output=True):
#         if not is_training:
#             return x
#         B, C, H, W = x.shape
#         x_masked = x.clone()
#         # 生成 Bernoulli 随机掩码 (1:保留, 0:遮挡)
#         mask = torch.bernoulli(torch.full((B, C, H, W), 1 - masking_prob)).to(x.device)
        
#         x_masked = x_masked * mask
        
#         if scale_output and masking_prob > 0:
#             x_masked /= (1 - masking_prob)
#         return x_masked

    def forward(self, x, need_fp = False, use_corr = False, ddp = None):
        """
        :param x2: c = 256
        :param x3: c = 512
        :param x4: c = 1024
        :param x5: c = 2048
        :return:
        """
        dict_return = {}
        h, w = x.shape[-2:]

        x1, x2, x3, x4, x5 = self.features(x)
        # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        # torch.Size([4, 64, 64, 64]) torch.Size([4, 256, 64, 64]) torch.Size([4, 512, 32, 32]) torch.Size([4, 1024, 16, 16]) torch.Size([4, 2048, 8, 8])
    
        # out_fp = self.decoder_ConvTranspose(dropout(x1),dropout(x2),dropout(x3),dropout(x4),dropout(x5)) # 43.08%
        # out_fp = self.decoder_ConvTranspose(dropout1(x1),dropout2(x2),dropout3(x3),dropout4(x4),dropout(x5)) # 44.20%
        dropout_layers = DynamicDropout(5, 0.1, 0.5, 1.0) 
        # for layer in range(1, 6):
        #     print(f"Layer {layer} Dropout Prob: {dropout_layers.get_dropout_prob(layer)}")
        out_fp, features2 = self.decoder_ConvTranspose(
            dropout_layers(x1, 1),
            dropout_layers(x2, 2),
            dropout_layers(x3, 3),  
            dropout_layers(x4, 4),  
            dropout_layers(x5, 5)   
        ) 
        dict_return['out_fp'] = out_fp
        dict_return['features2'] = features2
        # return dict_return

        # out = self.decoder_ConvTranspose(x1, x2, x3, x4, x5)
        # dict_return['out'] = out
        
        out, features1 = self.decoder_ConvTranspose(x1, x2, x3, x4, x5)
        dict_return['out'] = out
        dict_return['features1'] = features1

        return dict_return

class DynamicDropout(nn.Module):
    def __init__(self, num_layers: int = 5, p_min: float = 0.1, p_max: float = 0.5, alpha: float = 1.0):
        """
        多层动态增强 Dropout
        :param num_layers: 总层数
        :param p_min: 最小 Dropout 概率
        :param p_max: 最大 Dropout 概率
        :param alpha: 控制 Dropout 变化的指数因子
        """
        super(DynamicDropout, self).__init__()
        self.num_layers = num_layers
        self.p_min = p_min
        self.p_max = p_max
        self.alpha = alpha
        self.dropout_layers = nn.ModuleList([
            nn.Dropout2d(self.get_dropout_prob(layer_idx)) for layer_idx in range(1, num_layers + 1)
        ])
    def get_dropout_prob(self, layer_idx: int) :
        """计算第 layer_idx 层的 Dropout 概率"""
        prob = self.p_min + (self.p_max - self.p_min) * ((layer_idx - 1) / (self.num_layers - 1)) ** self.alpha
        return math.floor(prob * 100) / 100
    def forward(self, x, layer_idx: int):
        """
        前向传播
        :param x: 输入张量
        :param layer_idx: 当前所在层的索引 (从 1 开始)
        :return: 经过 Dropout 处理后的张量
        """
        return self.dropout_layers[layer_idx - 1](x)

class Corr(nn.Module):
    def __init__(self, nclass=15):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature_in, out):
        dict_return = {}
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))
        h_out, w_out = out.shape[2], out.shape[3]
        out = F.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)
        feature = F.interpolate(feature_in, (h_in, w_in), mode='bilinear', align_corners=True)
        f1 = rearrange(self.conv1(feature), 'n c h w -> n c (h w)')
        f2 = rearrange(self.conv2(feature), 'n c h w -> n c (h w)')
        out_temp = rearrange(out, 'n c h w -> n c (h w)')
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)
        dict_return['out'] = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)

        # corr_map_sample = self.sample(corr_map.detach(), h_in, w_in)
        # dict_return['corr_map'] = self.normalize_corr_map(corr_map_sample, h_in, w_in, h_out, w_out)

        return dict_return

    def sample(self, corr_map, h_in, w_in):
        index = torch.randint(0, h_in * w_in - 1, [128])
        corr_map_sample = corr_map[:, index.long(), :]
        return corr_map_sample

    def normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m, hw = corr_map.shape
        corr_map = rearrange(corr_map, 'n m (h w) -> (n m) 1 h w', h=h_in, w=w_in)
        corr_map = F.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)

        corr_map = rearrange(corr_map, '(n m) 1 h w -> (n m) (h w)', n=n, m=m)
        range_ = torch.max(corr_map, dim=1, keepdim=True)[0] - torch.min(corr_map, dim=1, keepdim=True)[0]
        temp_map = ((- torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / range_
        corr_map = (temp_map > 0.5)
        norm_corr_map = rearrange(corr_map, '(n m) (h w) -> n m h w', n=n, m=m, h=h_out, w=w_out)
        return norm_corr_map

def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


if __name__ == '__main__':
    net = se_resnext50_32x4d(12, pretrained=None)
    input = torch.rand([4, 3, 256, 256])
    # input.float().cuda()
    outs = net(input)
    # print(outs['out'].shape)
    # print(outs['out_fp'].shape)
    # print(outs['corr_out'].shape)