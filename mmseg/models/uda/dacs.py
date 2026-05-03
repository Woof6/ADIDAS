# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy
import torchvision.transforms as T

import mmcv
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio

# import torch_dct as dct

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
    
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            if isinstance(feat_imnet[0],list):
                feat_imnet = feat_imnet[0]
                feat = feat[0]
            feat_imnet = [f.detach() for f in feat_imnet]
        feat = [f.requires_grad_(True) for f in feat]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # t= torch.full((img.shape[0],),74, device = 'cuda', dtype = torch.long)
        # sample_t = 100
        # noise_gt= default(None, lambda: torch.randn([img.shape[0],img.shape[1],int(img.shape[2]/4),int(img.shape[3]/4)],requires_grad=False)).cuda()
        # noise = torch.nn.functional.interpolate(noise_gt, scale_factor=4, mode='bilinear', align_corners=False).cuda()
        # alpha_cumprod = torch.cumprod(1.-sigmoid_beta_schedule(sample_t), dim=0).cuda()
        # img_sample = extract(torch.sqrt( alpha_cumprod), t, img.shape) * img +extract(torch.sqrt(1- alpha_cumprod), t, img.shape) * noise
        
        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
   
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
                
        # target_img_sample = extract(torch.sqrt( alpha_cumprod), t, img.shape) * target_img +extract(torch.sqrt(1- alpha_cumprod), t, img.shape) * noise
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        
        # mixed_img_sample = extract(torch.sqrt( alpha_cumprod), t, img.shape) * mixed_img +extract(torch.sqrt(1- alpha_cumprod), t, img.shape) * noise
        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars




from mmseg.ops import resize
import torchgeometry as tgm

def build_mask_generator(cfg):
    if cfg is None:
        return None
    t = cfg.pop('type')
    if t == 'block':
        return BlockMaskGenerator(**cfg)
    else:
        raise NotImplementedError(t)


class BlockMaskGenerator:

    def __init__(self,  mask_block_size=32):
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs,mask_ratio):
        B, _, H, W = imgs.shape

        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > mask_ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs,mask_ratio):
        input_mask = self.generate_mask(imgs,mask_ratio)
        # data_2 = torch.rand((imgs.shape[0], imgs.shape[1]))*4 - 2
        # data_2 = data_2.unsqueeze(2)
        # data_2 = data_2.unsqueeze(3)
        # data_2 = data_2.expand(imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3])
        return imgs * input_mask #+ data_2.cuda()*(1-input_mask)



def exists(x):
    return x is not None
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
    
def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def block_noise(ref_x, randn_like=torch.randn_like, block_size=1, device=None):
    g_noise = randn_like(ref_x)
    if block_size == 1:
        return g_noise
    blk_noise = torch.zeros_like(ref_x, device=device)
    for px in range(block_size):
        for py in range(block_size):
            blk_noise += torch.roll(g_noise, shifts=(px, py), dims=(-2, -1))
            
    blk_noise = blk_noise / block_size # to maintain the same std on each pixel
    
    return blk_noise

@UDA.register_module()
class DACSt(UDADecorator):

    def __init__(self, **cfg):
        super(DACSt, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None
        
        
        # for name, module in self.get_model().named_modules():
        #     if isinstance(module, nn.Linear) and ("time" in name or "to_scale" in name):
        #         nn.init.constant_(module.weight,0)
        #         nn.init.constant_(module.bias,0)
        # self.mask_gen = BlockMaskGenerator()
        # self.gaussian_kernels = nn.ModuleList(self.get_kernels())
        # self.blur_layer = self.get_conv((33,33),(6,6))
        
    def get_conv(self, dims, std, mode='circular'):
        kernel = tgm.image.get_gaussian_kernel2d(dims, std)
        conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=dims, padding=int((dims[0]-1)/2), padding_mode=mode,
                         bias=False, groups=3)
        with torch.no_grad():
            kernel = torch.unsqueeze(kernel, 0)
            kernel = torch.unsqueeze(kernel, 0)
            kernel = kernel.repeat(3, 1, 1, 1)
            conv.weight = nn.Parameter(kernel)
        return conv

    def get_kernels(self):
        kernels = []
        for i in range(100):
            ks = 31
            kstd = np.exp(0.02 * i)
            # ks = 65
            # kstd = np.exp(0.03 * i)
            kernels.append(self.get_conv((ks, ks), (kstd, kstd), mode='reflect'))
        return kernels
    
    def q_sample(self, x_start, t):
        max_iters = torch.max(t)
        all_blurs = []
        x = x_start
        for i in range(max_iters+1):
            with torch.no_grad():
                x = self.gaussian_kernels[i](x)
                all_blurs.append(x)
        all_blurs = torch.stack(all_blurs)
        choose_blur = []
        # step is batch size as well so for the 49th step take the step(batch_size)
        for step in range(t.shape[0]):
            if step != -1:
                choose_blur.append(all_blurs[t[step], step])
            else:
                choose_blur.append(x_start[step])
        choose_blur = torch.stack(choose_blur)
        return choose_blur
    
    def get_mask_img(self,img,p):
        noise = torch.randn(img.shape,requires_grad=False).cuda()
        noise = self.blur_layer(noise)
        mean = torch.mean(noise[:,0:1])
        std = torch.std(noise[:,0:1])
        tau = mean + torch.sqrt(torch.tensor(2).float())*torch.erfinv((2*p-1).float())*std
        # print(mean,std,tau)
        mask = (noise[:,0:1]<tau.cuda()).float()
        # print(torch.sum(mask),(1-p)*512*512)
        return img*mask
        
        
    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
    
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        if isinstance(feat[0],list):
            feat = feat[0]
        feat = [f.requires_grad_(True) for f in feat]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
    

        # Train on source images
        sample_t = 100
        tau = 0.25
        tau2 = 5
        # sample_t1 = 100*(min(self.local_iter,40000)/40000)
        # sample_t1 = int(sample_t1)
        t = torch.randint(0, sample_t , (batch_size,),requires_grad=False).cuda().long()
      
        noise_gt= default(None, lambda: torch.randn([img.shape[0],img.shape[1],int(img.shape[2]/4),int(img.shape[3]/4)],requires_grad=False)).cuda()
        # noise_gt= default(None, lambda: torch.randn([img.shape[0],img.shape[1],int(img.shape[2]),int(img.shape[3])],requires_grad=False)).cuda()
        # noise_gt = block_noise(img,torch.randn_like,16,'cuda')
        noise = torch.nn.functional.interpolate(noise_gt, scale_factor=4, mode='bilinear', align_corners=False).cuda()
        # noise = noise_gt
        
        # noise_gt1 = torch.nn.functional.interpolate(img, scale_factor=0.25, mode='bilinear', align_corners=False).cuda()
        # noise_gt2 = torch.nn.functional.interpolate(target_img, scale_factor=0.25, mode='bilinear', align_corners=False).cuda()
        # noise_gt1 = torch.nn.functional.interpolate(target_img, scale_factor=0.25, mode='bilinear', align_corners=False).cuda()
        # noise_gt2 = torch.nn.functional.interpolate(torch.flip(img,[0]), scale_factor=0.25, mode='bilinear', align_corners=False).cuda()
        # noise1 = torch.nn.functional.interpolate(noise_gt1, scale_factor=4, mode='bilinear', align_corners=False).cuda()
        # noise2 = torch.nn.functional.interpolate(noise_gt2, scale_factor=4, mode='bilinear', align_corners=False).cuda()
        # noise_gt = default(None, lambda: torch.randn(img.shape,requires_grad=False)).cuda()
        # noise = noise_gt
      
        alpha_cumprod = torch.cumprod(1.-sigmoid_beta_schedule(sample_t), dim=0).cuda()
        # alpha_cumprod = torch.cumprod(1.-cosine_beta_schedule(sample_t), dim=0).cuda()
        # alpha_cumprod = torch.cumprod(1.-linear_beta_schedule(sample_t), dim=0).cuda()
        # ratio = torch.sqrt(1- alpha_cumprod)
        # print(alpha_cumprod,ratio)
        ## figure3 
        
        # snr = alpha_cumprod / (1 - alpha_cumprod)
        # snr_cl = torch.clamp(snr,max=5)
        # # snr_cl2 = torch.clamp(snr,min=1)
        # w1 = snr_cl/snr
        # w2 = snr/snr_cl2
        
        # img_sample = self.q_sample(img,t)
        # img_sample = torch.ones(img.shape, device=dev)
        # for i in range(batch_size):
        #     img_sample[i:i+1] = self.get_mask_img(img[i:i+1],alpha_cumprod[t[i].item()])
            # img_sample[i:i+1] = self.mask_gen.mask_image(img[i:i+1],ratio[t[i].item()])

        # img_ = torch.zeros_like(img).cuda()
        # for i in range(batch_size):    
        #     img_[i:i+1] , _ = strong_transform(
        #             strong_parameters,
        #             data=img[i:i+1].clone(),
        #             target=None)
        
        
        ##dct idea
        # img_sample_dct = torch.zeros_like(img)
        # img_sample = torch.zeros_like(img) 
        # for j in range(2):
        #     for i in range(3):
        #         img_sample_dct[j,i,:,:] = dct.dct_2d(img[j,i,:,:] , norm='ortho')
        #     img_sample_dct[j:j+1,:,:,:] = extract(torch.sqrt( alpha_cumprod), t[j:j+1,], img[j:j+1,:,:,:].shape) * img_sample_dct[j:j+1,:,:,:]  +extract(torch.sqrt(1- alpha_cumprod), t[j:j+1,], img[j:j+1,:,:,:].shape) * noise[j:j+1,:,:,:] 
        #     for i in range(3):
        #         img_sample[j,i,:,:] = dct.idct_2d(img_sample_dct[j,i,:,:] , norm='ortho')
        img_sample = extract(torch.sqrt( alpha_cumprod), t, img.shape) * img +extract(torch.sqrt(1- alpha_cumprod), t, img.shape) * noise
        
        clean_losses = self.get_model().forward_train(
                img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_losses = add_prefix(clean_losses, 'src')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss = (1-tau)*clean_loss
        clean_loss.backward(retain_graph=self.enable_fdist)
     
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')
        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,src_feat)
            feat_loss = (1-tau)*feat_loss
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
                
        if self.local_iter >=0:
            ### 加噪训练
            weight = torch.ones(
                [2,512,512], device=dev)
            weight[0]=weight[0]*tau#*w1[t[0].item()]#*np.exp(-t[0].item()/T)
            weight[1]=weight[1]*tau#*w1[t[1].item()]#np.exp(-t[1].item()/T)
            # for name, module in self.get_model().named_modules():
            #     if isinstance(module, nn.BatchNorm2d) and "fuse_layer." in name:
            #         module.training = False
            clean_losses = self.get_model().forward_train(
                [img_sample.float(),t], img_metas, [gt_semantic_seg,noise_gt],[weight,[tau2,tau2]], return_feat=True)

            # for name, module in self.get_model().named_modules():
            #     if isinstance(module, nn.BatchNorm2d) and "fuse_layer." in name:
            #         module.training = True
            src_feat = clean_losses.pop('features')
            clean_losses = add_prefix(clean_losses, 'src_denoise')
            clean_loss, clean_log_vars = self._parse_losses(clean_losses)
            log_vars.update(clean_log_vars)
            # for name, param in self.get_model().named_parameters():
            #     if "decode_head" in name and 'time' not in name and 'fuse_layer2' not in name and 'final_conv' not in name :
            #         param.requires_grad = False
            clean_loss.backward(retain_graph=self.enable_fdist)
            # for name, param in self.get_model().named_parameters():
            #     if "decode_head" in name and 'time' not in name and 'fuse_layer2' not in name and 'final_conv' not in name :
            #         param.requires_grad = True
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                seg_grads = [
                    p.grad.detach().clone() for p in params if p.grad is not None
                ]
                grad_mag = calc_grad_magnitude(seg_grads)
                mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')
            # ImageNet feature distance
            if self.enable_fdist:
                feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,src_feat)
                feat_loss = feat_loss*tau
                feat_loss.backward()
                log_vars.update(add_prefix(feat_log, 'src_denoise'))
                if self.print_grad_magnitude:
                    params = self.get_model().backbone.parameters()
                    fd_grads = [
                        p.grad.detach() for p in params if p.grad is not None
                    ]
                    fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                    grad_mag = calc_grad_magnitude(fd_grads)
                    mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

       
        ema_logits = self.get_ema_model().encode_decode(
             target_img, target_img_metas)
  
        
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        
        pseudo_weight_cpy = pseudo_weight
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)
        

        weight = torch.ones(
            [2,512,512], device=dev)
        weight[0]=pseudo_weight[0]*tau#*w1[t[0].item()]#*np.exp(-t[0].item()/T)
        weight[1]=pseudo_weight[1]*tau#*w1[t[1].item()]#*np.exp(-t[1].item()/T)
        
       
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
            # ## 搞一下
            # pseudo_weight[:, :, :self.psweight_ignore_top] = 0
            # pseudo_weight[:, :,-self.psweight_ignore_top :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)   
         
        # weight = torch.ones(
        #     [2,512,512], device=dev)
        # weight[0]=pseudo_weight[0]*tau#*w1[t[0].item()]#*np.exp(-t[0].item()/T)
        # weight[1]=pseudo_weight[1]*tau#*w1[t[1].item()]#*np.exp(-t[1].item()/T)    
            
        

        # weight = torch.ones(
        #     [2,512,512], device=dev)
        # weight[0]=pseudo_weight[0]*tau#*w1[t[0].item()]#*np.exp(-t[0].item()/T)
        # weight[1]=pseudo_weight[1]*tau#*w1[t[1].item()]#*np.exp(-t[1].item()/T)
        
        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)


                
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))

        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=False)
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss = (1-tau)*mix_loss
        mix_loss.backward()

        

       
        
        
        # t2 = torch.randint(0, sample_t , (batch_size,),requires_grad=False).cuda().long()
        # noise_gt2= default(None, lambda: torch.randn([img.shape[0],img.shape[1],int(img.shape[2]/4),int(img.shape[3]/4)],requires_grad=False)).cuda()
        # noise2 = torch.nn.functional.interpolate(noise_gt2, scale_factor=4, mode='bilinear', align_corners=False).cuda()
        ###加噪
        # target_img_sample = self.q_sample(target_img,t)
        # target_img_sample = torch.ones(img.shape, device=dev)
        # for i in range(batch_size):
        #     target_img_sample[i:i+1] = self.get_mask_img(target_img[i:i+1],alpha_cumprod[t[i].item()])
            # target_img_sample[i:i+1] = self.mask_gen.mask_image(target_img[i:i+1],ratio[t[i].item()])
        # strong_parameters['mix'] = None
        # for i in range(batch_size):    
        #     target_img[i:i+1] , _ = strong_transform(
        #             strong_parameters,
        #             data=target_img[i:i+1],
        #             target=None)
        
        
        # target_img_sample_dct = torch.zeros_like(img)
        # target_img_sample = torch.zeros_like(img) 
        # for j in range(2):
        #     for i in range(3):
        #         target_img_sample_dct[j,i,:,:] = dct.dct_2d(target_img[j,i,:,:] , norm='ortho')
        #     target_img_sample_dct[j:j+1,:,:,:] = extract(torch.sqrt( alpha_cumprod), t[j:j+1,], img[j:j+1,:,:,:].shape) * target_img_sample_dct[j:j+1,:,:,:]  +extract(torch.sqrt(1- alpha_cumprod), t[j:j+1,], img[j:j+1,:,:,:].shape) * noise[j:j+1,:,:,:]     
        #     for i in range(3):
        #         target_img_sample[j,i,:,:] = dct.idct_2d(target_img_sample_dct[j,i,:,:] , norm='ortho')
        target_img_sample = extract(torch.sqrt(alpha_cumprod), t, target_img.shape) * target_img +extract(torch.sqrt(1-alpha_cumprod), t, target_img.shape) * noise
        
        # for name, module in self.get_model().named_modules():
        #     if isinstance(module, nn.BatchNorm2d) and "fuse_layer." in name:
        #         module.training = False
        if self.local_iter >=0:
            mix_losses = self.get_model().forward_train(
                [target_img_sample.float(),t], img_metas, [pseudo_label.unsqueeze(1),noise_gt], [weight,[tau2,tau2]], return_feat=False)
            # mix_losses = self.get_model().forward_train(
            #     [target_img_sample.float(),t], img_metas, [mixed_lbl,noise_gt], [pseudo_weight*tau,[tau2,tau2]], return_feat=False)
            # mix_losses = self.get_model().forward_train(
            #     target_img_sample.float(), img_metas, pseudo_label.unsqueeze(1), weight, return_feat=False)
            # for name, module in self.get_model().named_modules():
            #     if isinstance(module, nn.BatchNorm2d) and "fuse_layer." in name:
            #         module.training = True
            mix_losses = add_prefix(mix_losses, 'target_denoise')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            # for name, param in self.get_model().named_parameters():
            #     if "decode_head" in name and 'time' not in name and 'fuse_layer2' not in name and 'final_conv' not in name :
            #         param.requires_grad = False
            mix_loss.backward()
            # for name, param in self.get_model().named_parameters():
            #     if "decode_head" in name and 'time' not in name and 'fuse_layer2' not in name and 'final_conv' not in name :
            #         param.requires_grad =True
        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            vis_img_sample = torch.clamp(denorm(img_sample, means, stds), 0, 1)
            vis_mixed_img_sample = torch.clamp(denorm(target_img_sample, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 6
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                    
                subplotimg(axs[0][5], vis_img_sample[j], 'sample'+str(int(t[j].item()))+'_Source Image')
                subplotimg(axs[1][5], vis_mixed_img_sample[j], 'sample'+str(int(t[j].item()))+'_Target Image')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars




















# ## figure3 
# img_sample_25 = self.q_sample(img,torch.full((img.shape[0],),24, device = 'cuda', dtype = torch.long))
# img_sample_50 = self.q_sample(img,torch.full((img.shape[0],),49, device = 'cuda', dtype = torch.long))
# img_sample_75 = self.q_sample(img,torch.full((img.shape[0],),74, device = 'cuda', dtype = torch.long))
# img_sample_100 = self.q_sample(img,torch.full((img.shape[0],),99, device = 'cuda', dtype = torch.long))
# target_img_sample_25 = self.q_sample(target_img,torch.full((img.shape[0],),24, device = 'cuda', dtype = torch.long))
# target_img_sample_50 = self.q_sample(target_img,torch.full((img.shape[0],),49, device = 'cuda', dtype = torch.long))
# target_img_sample_75 = self.q_sample(target_img,torch.full((img.shape[0],),74, device = 'cuda', dtype = torch.long))
# target_img_sample_100 = self.q_sample(target_img,torch.full((img.shape[0],),99, device = 'cuda', dtype = torch.long))
# if self.local_iter % self.debug_img_interval == 0:
#     out_dir = os.path.join(self.train_cfg['work_dir'],
#                             'class_mix_debug')
#     os.makedirs(out_dir, exist_ok=True)
#     vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
#     vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
#     vis_img_25 = torch.clamp(denorm(img_sample_25, means, stds), 0, 1)
#     vis_trg_img_25 = torch.clamp(denorm(target_img_sample_25, means, stds), 0, 1)
#     vis_img_50 = torch.clamp(denorm(img_sample_50, means, stds), 0, 1)
#     vis_trg_img_50 = torch.clamp(denorm(target_img_sample_50, means, stds), 0, 1)
#     vis_img_75 = torch.clamp(denorm(img_sample_75, means, stds), 0, 1)
#     vis_trg_img_75 = torch.clamp(denorm(target_img_sample_75, means, stds), 0, 1)
#     vis_img_100 = torch.clamp(denorm(img_sample_100, means, stds), 0, 1)
#     vis_trg_img_100 = torch.clamp(denorm(target_img_sample_100, means, stds), 0, 1)
#     for j in range(batch_size):
#         rows, cols = 2, 5
#         fig, axs = plt.subplots(
#             rows,
#             cols,
#             figsize=(3 * cols, 3 * rows),
#             gridspec_kw={
#                 'hspace': 0.1,
#                 'wspace': 0,
#                 'top': 0.95,
#                 'bottom': 0,
#                 'right': 1,
#                 'left': 0
#             },
#         )
#         subplotimg(axs[0][0], vis_img[j], 'Source Image')
#         subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
#         subplotimg(axs[0][1], vis_img_25[j], 'Source Image')
#         subplotimg(axs[1][1], vis_trg_img_25[j], 'Target Image')
#         subplotimg(axs[0][2], vis_img_50[j], 'Source Image')
#         subplotimg(axs[1][2], vis_trg_img_50[j], 'Target Image')
#         subplotimg(axs[0][3], vis_img_75[j], 'Source Image')
#         subplotimg(axs[1][3], vis_trg_img_75[j], 'Target Image')
#         subplotimg(axs[0][4], vis_img_100[j], 'Source Image')
#         subplotimg(axs[1][4], vis_trg_img_100[j], 'Target Image')
#         for ax in axs.flat:
#             ax.axis('off')
#         plt.savefig(
#             os.path.join(out_dir,
#                             f'blur{(self.local_iter + 1):06d}_{j}.png'))
#         plt.close()
# img_sample_25 = torch.ones(img.shape, device=dev)
# img_sample_50 = torch.ones(img.shape, device=dev)
# img_sample_75 = torch.ones(img.shape, device=dev)
# img_sample_100 = torch.ones(img.shape, device=dev)
# target_img_sample_25 = torch.ones(img.shape, device=dev)
# target_img_sample_50 = torch.ones(img.shape, device=dev)
# target_img_sample_75 = torch.ones(img.shape, device=dev)
# target_img_sample_100 = torch.ones(img.shape, device=dev)
# for i in range(batch_size):
#     img_sample_25[i:i+1] = self.get_mask_img(img[i:i+1],alpha_cumprod[24])
#     img_sample_50[i:i+1] = self.get_mask_img(img[i:i+1],alpha_cumprod[49])
#     img_sample_75[i:i+1] = self.get_mask_img(img[i:i+1],alpha_cumprod[74])
#     img_sample_100[i:i+1] = self.get_mask_img(img[i:i+1],alpha_cumprod[99])
#     target_img_sample_25[i:i+1] = self.get_mask_img(target_img[i:i+1],alpha_cumprod[24])
#     target_img_sample_50[i:i+1] = self.get_mask_img(target_img[i:i+1],alpha_cumprod[49])
#     target_img_sample_75[i:i+1] = self.get_mask_img(target_img[i:i+1],alpha_cumprod[74])
#     target_img_sample_100[i:i+1] = self.get_mask_img(target_img[i:i+1],alpha_cumprod[99])
# vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
# vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
# vis_img_25 = torch.clamp(denorm(img_sample_25, means, stds), 0, 1)
# vis_trg_img_25 = torch.clamp(denorm(target_img_sample_25, means, stds), 0, 1)
# vis_img_50 = torch.clamp(denorm(img_sample_50, means, stds), 0, 1)
# vis_trg_img_50 = torch.clamp(denorm(target_img_sample_50, means, stds), 0, 1)
# vis_img_75 = torch.clamp(denorm(img_sample_75, means, stds), 0, 1)
# vis_trg_img_75 = torch.clamp(denorm(target_img_sample_75, means, stds), 0, 1)
# vis_img_100 = torch.clamp(denorm(img_sample_100, means, stds), 0, 1)
# vis_trg_img_100 = torch.clamp(denorm(target_img_sample_100, means, stds), 0, 1)
# for j in range(batch_size):
#         rows, cols = 2, 5
#         fig, axs = plt.subplots(
#             rows,
#             cols,
#             figsize=(3 * cols, 3 * rows),
#             gridspec_kw={
#                 'hspace': 0.1,
#                 'wspace': 0,
#                 'top': 0.95,
#                 'bottom': 0,
#                 'right': 1,
#                 'left': 0
#             },
#         )
#         subplotimg(axs[0][0], vis_img[j], 'Source Image')
#         subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
#         subplotimg(axs[0][1], vis_img_25[j], 'Source Image')
#         subplotimg(axs[1][1], vis_trg_img_25[j], 'Target Image')
#         subplotimg(axs[0][2], vis_img_50[j], 'Source Image')
#         subplotimg(axs[1][2], vis_trg_img_50[j], 'Target Image')
#         subplotimg(axs[0][3], vis_img_75[j], 'Source Image')
#         subplotimg(axs[1][3], vis_trg_img_75[j], 'Target Image')
#         subplotimg(axs[0][4], vis_img_100[j], 'Source Image')
#         subplotimg(axs[1][4], vis_trg_img_100[j], 'Target Image')
#         for ax in axs.flat:
#             ax.axis('off')
#         plt.savefig(
#             os.path.join(out_dir,
#                             f'mask{(self.local_iter + 1):06d}_{j}.png'))
#         plt.close()