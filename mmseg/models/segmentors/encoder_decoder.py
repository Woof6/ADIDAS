# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
import numpy as np
from matplotlib import pyplot as plt
from mmseg.models.utils.visualization import subplotimg
import torchgeometry as tgm

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
def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0
def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.gaussian_kernels = nn.ModuleList(self.get_kernels())
        # self.blur_layer = self.get_conv((33,33),(6,6))
        
        assert self.with_decode_head
        
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
            
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
 
            
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        if isinstance(out,list):
            out[0] = resize(
                input=out[0],
                size=img[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        else:
            out= resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
      
        return out

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        # if isinstance(seg_logits,list):
        #     seg_logits = seg_logits[0]
        return seg_logits

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      seg_weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False):
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
        
        ## 加噪
        
        # t= torch.full((img.shape[0],),74, device = 'cuda', dtype = torch.long)
        # sample_t = 100
        # noise_gt= default(None, lambda: torch.randn([img.shape[0],img.shape[1],int(img.shape[2]/4),int(img.shape[3]/4)],requires_grad=False)).cuda()
        # noise = torch.nn.functional.interpolate(noise_gt, scale_factor=4, mode='bilinear', align_corners=False).cuda()
        # alpha_cumprod = torch.cumprod(1.-sigmoid_beta_schedule(sample_t), dim=0).cuda()
        # img = extract(torch.sqrt( alpha_cumprod), t, img.shape) * img +extract(torch.sqrt(1- alpha_cumprod), t, img.shape) * noise
        x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
    
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        # h_stride, w_stride = 256,256
        # h_crop, w_crop = 512,512
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
     
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds
    
    
    
    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        
        seg_logit = self.encode_decode(img, img_meta)
       
        # means, stds = get_mean_std(img_meta, img.device)
        # t1 = torch.full((img.shape[0],),24, device = 'cuda', dtype = torch.long)
        # t2= torch.full((img.shape[0],),49, device = 'cuda', dtype = torch.long)
        # t3= torch.full((img.shape[0],),74, device = 'cuda', dtype = torch.long)
        # t4= torch.full((img.shape[0],),99, device = 'cuda', dtype = torch.long)
        # sample_t = 100
        # noise_gt= default(None, lambda: torch.randn([img.shape[0],img.shape[1],int(img.shape[2]/4),int(img.shape[3]/4)],requires_grad=False)).cuda()
        # noise = torch.nn.functional.interpolate(noise_gt, scale_factor=4, mode='bilinear', align_corners=False).cuda()
        # alpha_cumprod = torch.cumprod(1.-sigmoid_beta_schedule(sample_t), dim=0).cuda()
        # img_sample1 = extract(torch.sqrt( alpha_cumprod), t1, img.shape) * img +extract(torch.sqrt(1- alpha_cumprod), t1, img.shape) * noise
        # # img_sample1 = self.q_sample(img,t1)
        # # img_sample1 = torch.ones(img.shape, device=img.device)
        # # img_sample1[0:1] = self.get_mask_img(img[0:1],alpha_cumprod[t1[0].item()])
        # img_sample1 = img_sample1.float()
        # img_sample2 = extract(torch.sqrt( alpha_cumprod), t2, img.shape) * img +extract(torch.sqrt(1- alpha_cumprod), t2, img.shape) * noise
        # # img_sample2 = self.q_sample(img,t2)
        # # img_sample2 = torch.ones(img.shape, device=img.device)
        # # img_sample2[0:1] = self.get_mask_img(img[0:1],alpha_cumprod[t2[0].item()])
        # img_sample2 = img_sample2.float()
        # img_sample3 = extract(torch.sqrt( alpha_cumprod), t3, img.shape) * img +extract(torch.sqrt(1- alpha_cumprod), t3, img.shape) * noise
        # # img_sample3 = self.q_sample(img,t3)
        # # img_sample3 = torch.ones(img.shape, device=img.device)
        # # img_sample3[0:1] = self.get_mask_img(img[0:1],alpha_cumprod[t3[0].item()])
        # img_sample3 = img_sample3.float()
        # img_sample4 = extract(torch.sqrt( alpha_cumprod), t4, img.shape) * img +extract(torch.sqrt(1- alpha_cumprod), t3, img.shape) * noise
        # # img_sample4 = self.q_sample(img,t4)
        # # img_sample4 = torch.ones(img.shape, device=img.device)
        # # img_sample4[0:1] = self.get_mask_img(img[0:1],alpha_cumprod[t4[0].item()])
        # img_sample4 = img_sample4.float()
        
        
        
        # vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
        # vis_img_sample1 = torch.clamp(denorm(img_sample1 , means, stds), 0, 1)
        # vis_img_sample2 = torch.clamp(denorm(img_sample2 , means, stds), 0, 1)
        # vis_img_sample3 = torch.clamp(denorm(img_sample3 , means, stds), 0, 1)
        # vis_img_sample4 = torch.clamp(denorm(img_sample4 , means, stds), 0, 1)
        
        # rows, cols = 5, 5
        # fig, axs = plt.subplots(
        #     rows,
        #     cols,
        #     figsize=(6* cols, 3 * rows),
        #     gridspec_kw={
        #         'hspace': 0.1,
        #         'wspace': 0,
        #         'top': 0.95,
        #         'bottom': 0,
        #         'right': 1,
        #         'left': 0
        #     },
        # )
        # subplotimg(axs[0][0], vis_img[0],'1')
        # subplotimg(axs[0][1], vis_img_sample1[0],'1')
        # subplotimg(axs[0][2], vis_img_sample2[0],'1')
        # subplotimg(axs[0][3], vis_img_sample3[0],'1')
        # subplotimg(axs[0][4], vis_img_sample4[0],'1')
      
        # for i in range(4):
        #     t = torch.full((img.shape[0],),25*i+24, device = 'cuda', dtype = torch.long)
        #     seg_logit = self.encode_decode([img,t], img_meta)
        #     x_start =  extract(torch.sqrt(1. / alpha_cumprod), t, img.shape) * img -extract(torch.sqrt(1. / alpha_cumprod - 1),t, img.shape) * torch.nn.functional.interpolate(seg_logit[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
        #     # x_start =  torch.nn.functional.interpolate(seg_logit[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
        #     seg_logit = self.encode_decode([img_sample1,t], img_meta)
        #     x_start1 =  extract(torch.sqrt(1. / alpha_cumprod), t, img.shape) * img_sample1 -extract(torch.sqrt(1. / alpha_cumprod - 1),t, img.shape) * torch.nn.functional.interpolate(seg_logit[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
        #     # x_start1 =  torch.nn.functional.interpolate(seg_logit[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
        #     seg_logit = self.encode_decode([img_sample2,t], img_meta)
        #     x_start2 =  extract(torch.sqrt(1. / alpha_cumprod), t, img.shape) * img_sample2 -extract(torch.sqrt(1. / alpha_cumprod - 1),t, img.shape) * torch.nn.functional.interpolate(seg_logit[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
        #     # x_start2 =  torch.nn.functional.interpolate(seg_logit[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
        #     seg_logit = self.encode_decode([img_sample3,t], img_meta)
        #     x_start3 =  extract(torch.sqrt(1. / alpha_cumprod), t, img.shape) * img_sample3 -extract(torch.sqrt(1. / alpha_cumprod - 1),t, img.shape) * torch.nn.functional.interpolate(seg_logit[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
        #     # x_start3 =  torch.nn.functional.interpolate(seg_logit[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
        #     seg_logit = self.encode_decode([img_sample4,t], img_meta)
        #     x_start4 =  extract(torch.sqrt(1. / alpha_cumprod), t, img.shape) * img_sample4 -extract(torch.sqrt(1. / alpha_cumprod - 1),t, img.shape) * torch.nn.functional.interpolate(seg_logit[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
        #     # x_start4 =  torch.nn.functional.interpolate(seg_logit[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
        #     x_start = torch.clamp(denorm(x_start, means, stds), 0, 1)
        #     x_start1 = torch.clamp(denorm(x_start1 , means, stds), 0, 1)
        #     x_start2 = torch.clamp(denorm(x_start2 , means, stds), 0, 1)
        #     x_start3 = torch.clamp(denorm(x_start3 , means, stds), 0, 1)
        #     x_start4 = torch.clamp(denorm(x_start4 , means, stds), 0, 1)
        #     # print(t)
        #     # print(torch.mean(torch.square(x_start2-img_sample2)))
        #     subplotimg(axs[i+1][0], x_start[0],str(i))
        #     subplotimg(axs[i+1][1], x_start1[0],str(i))
        #     subplotimg(axs[i+1][2], x_start2[0],str(i))
        #     subplotimg(axs[i+1][3], x_start3[0],str(i))
        #     subplotimg(axs[i+1][4], x_start4[0],str(i))
        # for ax in axs.flat:
        #      ax.axis('off')
        # plt.savefig('/ssd/lwk/DAFormer/demo/demo_fig.png')
        # plt.close()
        # seg_logit = self.encode_decode(x_start1.float(), img_meta)
  
       
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
