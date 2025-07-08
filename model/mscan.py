import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple

from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(BaseModule):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class MSCAN(BaseModule):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[32, 64, 160, 256],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 3, 5, 2],
                 num_stages=4,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None,
                 num_classes=12):
           
        super(MSCAN, self).__init__(init_cfg=init_cfg)
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        # if isinstance(pretrained, str):
        #     warnings.warn('DeprecationWarning: pretrained is deprecated, '
        #                   'please use "init_cfg" instead')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        # elif pretrained is not None:
        #     raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_cfg=norm_cfg)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(320, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(512, 320, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(320)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        # self.relu = nn.ReLU(inplace=True)
        # self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.deconv4 = nn.ConvTranspose2d(160, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.deconv5 = nn.ConvTranspose2d(256, 160, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn5 = nn.BatchNorm2d(160)
        # self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(MSCAN, self).init_weights()

    def features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs
    
    def decoder_ConvTranspose(self, x1, x2, x3, x4):
        """
        :rtype: object
        """
        score = self.bn5(self.relu(self.deconv5(x4))) 
        score = score + x3  # b 320 20 20
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = score + x2  # b 128 40 40
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = score + x1  # b 64 80 80
        score = self.bn2(self.relu(self.deconv2(score)))  # b 64 160 160
        # if refined_features != None:
        #     score = torch.cat((score, F.interpolate(refined_features[-1],score.shape[-2:], mode='bilinear')),dim=1)
        score = self.bn1(self.relu(self.deconv1(score)))  # b 64 320 320 

        seg_out = self.classifier(score)
        return seg_out
    
    def forward(self, x):
        dict_return = {}
        h, w = x.shape[-2:]

        x1, x2, x3, x4 = self.features(x)
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        
        # torch.Size([4, 64, 80, 80]) torch.Size([4, 128, 40, 40]) torch.Size([4, 320, 20, 20]) torch.Size([4, 512, 10, 10])
        # torch.Size([4, 64, 80, 80]) torch.Size([4, 128, 40, 40]) torch.Size([4, 320, 20, 20]) torch.Size([4, 512, 10, 10])
        
        out = self.decoder_ConvTranspose(x1, x2, x3, x4)
        dict_return['out'] = out
        
        return dict_return

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6
    
    
def main():
    # mscan_s配置参数
    # embed_dims = [64, 128, 320, 512]  # 每一层的嵌入维度
    # depths = [2, 2, 4, 2]  # 每一层的深度
    # norm_cfg = dict(type='BN', requires_grad=True)
    # init_cfg = dict(type='Pretrained', checkpoint='../pretrained/mscan_s.pth')
    
    # mscan_b配置参数
    # embed_dims = [64, 128, 320, 512]  # 每一层的嵌入维度
    # depths = [3, 3, 12, 3]  # 每一层的深度
    # norm_cfg = dict(type='BN', requires_grad=True)
    # init_cfg = dict(type='Pretrained', checkpoint='../pretrained/mscan_b.pth')
    # drop_path_rate=0.1
    
    # mscan_l配置参数
    # embed_dims = [64, 128, 320, 512]  # 每一层的嵌入维度
    # depths = [3, 5, 27, 3]  # 每一层的深度
    # norm_cfg = dict(type='BN', requires_grad=True)
    # init_cfg = dict(type='Pretrained', checkpoint='../pretrained/mscan_l.pth')
    # drop_path_rate=0.3

    # 创建 MSCAN 模型
    # model = MSCAN(
    #     embed_dims=embed_dims,
    #     depths=depths,
    #     norm_cfg=norm_cfg,
    #     init_cfg=init_cfg,
    #     drop_path_rate=drop_path_rate
    # ).cuda()
    
    # mscan_t配置参数
    init_cfg = dict(type='Pretrained', checkpoint='../pretrained/mscan_t.pth')
    model = MSCAN(init_cfg=init_cfg).cuda()
    
    model.init_weights()

    model = nn.DataParallel(model).cuda()

    print("total: ", count_params(model))

    # 模拟输入：一个随机生成的批次数据（batch size 1, 3 通道, 高度 224, 宽度 224）
    x = torch.randn(8, 3, 320, 320)

    # 前向传播
    output = model(x)

    # 打印输出的形状
    print("Output shape:", output['out'].shape)


if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    print(f"当前可用的 GPU 数量: {gpu_count}")
    main()