# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 time_dim = None,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg,
            time_dim = None)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        if time_dim is not None:
            self.time_mlp=nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, (len(dilations) + int(pool) + int(bool(context_cfg))) * channels * 2)
            ) 
            # self.time_mlp2=nn.Sequential(
            #     nn.SiLU(),
            #     nn.Linear(time_dim,  channels * 2)
            # ) 
         
            
    def forward(self, x,t=None):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        if t is not None:
            t_ = self.time_mlp(t)
            time_emb = rearrange(t_, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
            scale, shift = scale_shift
            aspp_outs = aspp_outs * (scale + 1) + shift
            # aspp_outs = aspp_outs + shift
            
        output = self.bottleneck(aspp_outs)
        
        # if t is not None:
        #     t = self.time_mlp2(t)
        #     time_emb = rearrange(t, 'b c -> b c 1 1')
        #     scale_shift = time_emb.chunk(2, dim = 1)
        #     scale, shift = scale_shift
        #     output= output * (scale + 1) + shift
        
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


@HEADS.register_module()
class DAFormerHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(DAFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        x = self.cls_seg(x)

        return x



from einops import rearrange, repeat, reduce, pack, unpack

    
@HEADS.register_module()
class DAFormerHeadt(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(DAFormerHeadt, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        
        time_dim = 256
        self.embed_layers = {}
        self.time_emb_layers ={}
        # self.embed_layers2 ={}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
                # self.embed_layers2[str(i)] = build_layer(
                #     in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
                # self.embed_layers2[str(i)] = build_layer(
                #     in_channels, embed_dim, **embed_cfg)
                
            self.time_emb_layers[str(i)] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim , embed_dim * 2)
            ) 
        self.time_emb_layers = nn.ModuleDict(self.time_emb_layers)
        self.embed_layers = nn.ModuleDict(self.embed_layers)
        # self.embed_layers2 = nn.ModuleDict(self.embed_layers2)
 
        
    
        fusion_cfg['time_dim'] = time_dim 
        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)
        
        self.fuse_layer1 = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)
        
        
        # fusion_cfg =dict(
        #             type='conv',
        #             kernel_size=1,
        #             act_cfg=dict(type='ReLU'),
        #             norm_cfg=fusion_cfg['norm_cfg'])
        self.fuse_layer2 = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)
        self.final_conv = nn.Conv2d(self.channels, 3, 1)
        # self.final_conv = nn.ConvTranspose2d(self.channels, 3,4,4)
        # self.ps = nn.PixelShuffle(4)
        # self.cls_seg2 = nn.Sequential(nn.Dropout2d(0.1),nn.Conv2d(self.channels, 19, 1))


    def forward(self, inputs):
        x = inputs
        if isinstance(inputs[0],list):
            n, _, h, w = x[0][-1].shape
            os_size = x[0][0].size()[2:]
            _c = {}
            for i in self.in_index:
                _c[i] = self.embed_layers[str(i)](x[0][i])
                time_emb = self.time_emb_layers[str(i)](x[1])
                time_emb = rearrange(time_emb, 'b c -> b c 1 1')
                scale_shift = time_emb.chunk(2, dim = 1)
                scale, shift = scale_shift
                if _c[i].dim() == 3:
                    _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                        .reshape(n, -1, x[0][i].shape[2], x[0][i].shape[3])
                if _c[i].size()[2:] != os_size:
                    _c[i] = resize(
                        _c[i],
                        size=os_size,
                        mode='bilinear',
                        align_corners=self.align_corners)
                _c[i] = _c[i] * (scale + 1) + shift
                # _c[i] = _c[i] + shift
            #head3 
            x2 = self.fuse_layer2(torch.cat(list(_c.values()),dim=1),x[1])
            noise =self.final_conv(x2)
            x = self.fuse_layer1(torch.cat(list(_c.values()),dim=1),x[1])
            x = self.cls_seg(x)
            
            # head2
            # x2 = self.fuse_layer2(torch.cat(list(_c.values()),dim=1),x[1])
            # noise =self.final_conv(x2)
            # x = self.fuse_layer(torch.cat(list(_c.values()), dim=1),x[1])
            # x = self.cls_seg(x)
            
            # head0
            # x = self.fuse_layer2(torch.cat(list(_c.values()),dim=1))
            # noise =self.final_conv(x)
            # x = self.cls_seg(x)

            
            #super fuse
            # x2 = self.fuse_layer2(torch.cat(list(_c.values()),dim=1),x[1])  
            # noise =self.final_conv(x2)
            # x = self.fuse_layer1(torch.cat(list(_c.values())+[x2],dim=1),x[1])
            # x = self.cls_seg2(x)

        
            # return [x,self.ps(noise)]
            return [x,noise]
        else:
            n, _, h, w = x[-1].shape
            os_size = x[0].size()[2:]
            _c = {}
            for i in self.in_index:
                _c[i] = self.embed_layers[str(i)](x[i])
                if _c[i].dim() == 3:
                    _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                        .reshape(n, -1, x[i].shape[2], x[i].shape[3])
                if _c[i].size()[2:] != os_size:
                    _c[i] = resize(
                        _c[i],
                        size=os_size,
                        mode='bilinear',
                        align_corners=self.align_corners)
            x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
            x = self.cls_seg(x)
            return x
