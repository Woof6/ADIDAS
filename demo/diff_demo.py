import os
from argparse import ArgumentParser

import mmcv
from tools.test import update_legacy_cfg
import torch.nn.functional as F
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette
import torch
import cv2
from torchvision import transforms
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
from tqdm.auto import tqdm


T = transforms.ToPILImage()

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


def sample(model,sample_t):
   
    img = torch.randn([1,3,128,128], device = 'cuda')
    img = torch.nn.functional.interpolate(img, scale_factor=4, mode='bilinear', align_corners=False).cuda()
 
    betas = sigmoid_beta_schedule(sample_t).cuda()
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).cuda()
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_log_variance_clipped= torch.log(posterior_variance.clamp(min =1e-20))
   
    # for t in tqdm(reversed(range(0, sample_t)), desc = 'sampling loop time step', total = sample_t):
        
    #     batched_times = torch.full((1,), t, device = 'cuda', dtype = torch.long)
    #     with torch.no_grad(): 
    #         noise_pred =  model.encode_decode([img.float(),batched_times],None)
    #     noise_pred_up = torch.nn.functional.interpolate(noise_pred[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
    #     x_start =  extract(torch.sqrt(1. / alphas_cumprod), batched_times, img.shape) * img -extract(torch.sqrt(1. / alphas_cumprod - 1),batched_times, img.shape) * noise_pred_up
    #     # x_start.clamp_(-1., 1.)
    #     model_mean= (
    #         extract(betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), batched_times, img.shape) * x_start +
    #         extract( (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod), batched_times, img.shape) * img
    #     )
    #     posterior_log_variance = extract(posterior_log_variance_clipped, batched_times, img.shape)
    #     noise = torch.nn.functional.interpolate(torch.randn([1,3,128,128], device = 'cuda'), scale_factor=4, mode='bilinear', align_corners=False).cuda() if t > 0 else 0. # no noise if t == 0
    #     img = model_mean  + (0.5 * posterior_log_variance).exp() * noise
    #     print(torch.max(img),torch.min(img))
    # for t in tqdm(reversed(range(0, sample_t)), desc = 'sampling loop time step', total = sample_t):
        
    #     batched_times = torch.full((1,), t, device = 'cuda', dtype = torch.long)
    #     with torch.no_grad(): 
    #         noise_pred =  model.encode_decode([img.float(),batched_times],None)
    #     img = torch.nn.functional.interpolate(img, scale_factor=0.25, mode='bilinear', align_corners=False).cuda()
    #     x_start =  extract(torch.sqrt(1. / alphas_cumprod), batched_times, img.shape) * img -extract(torch.sqrt(1. / alphas_cumprod - 1),batched_times, img.shape) * noise_pred[1]
    #     #x_start.clamp_(-1., 1.)
    #     model_mean= (
    #         extract(betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), batched_times, img.shape) * x_start +
    #         extract( (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod), batched_times, img.shape) * img
    #     )
    #     posterior_log_variance = extract(posterior_log_variance_clipped, batched_times, img.shape)
    #     noise = torch.randn([1,3,128,128]).cuda() if t > 0 else 0. # no noise if t == 0
    #     img = model_mean  + (0.5 * posterior_log_variance).exp() * noise
    #     print(torch.max(img),torch.min(img))
    #     img = torch.nn.functional.interpolate(img, scale_factor=4, mode='bilinear', align_corners=False).cuda()
    img = torch.randn([1,3,512,512], device = 'cuda')
    for t in tqdm(reversed(range(0, sample_t)), desc = 'sampling loop time step', total = sample_t):
        batched_times = torch.full((1,), t, device = 'cuda', dtype = torch.long)
        with torch.no_grad(): 
            noise_pred =  model.encode_decode([img.float(),batched_times],None)
        x_start =  extract(torch.sqrt(1. / alphas_cumprod), batched_times, img.shape) * img -extract(torch.sqrt(1. / alphas_cumprod - 1),batched_times, img.shape) * noise_pred[1]
        x_start.clamp_(-1., 1.)
        model_mean= (
            extract(betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), batched_times, img.shape) * x_start +
            extract( (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod), batched_times, img.shape) * img
        )
        posterior_log_variance = extract(posterior_log_variance_clipped, batched_times, img.shape)
        noise = torch.randn([1,3,512,512]).cuda() if t > 0 else 0. # no noise if t == 0
        img = model_mean  + (0.5 * posterior_log_variance).exp() * noise
        print(torch.max(img),torch.min(img))
        print(torch.max(noise_pred[1]),torch.min(noise_pred[1]))
      
    npimg = img[0].cpu()
    print(torch.max(npimg),torch.min(npimg))
    npimg = (npimg+1)/2
    npimg = torch.clamp(npimg , 0, 1)
    npimg = T(npimg)
    npimg.save('demo/demo_sample.jpg')
    
def main():
    parser = ArgumentParser()

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile('./work_dirs/local-basic/230506_1551_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_4x127.5_14f2f/230506_1551_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_ddpm_14f2f.json')
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        './work_dirs/local-basic/230506_1551_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_4x127.5_14f2f/latest.pth',
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
    sample_t =100
    sample(model,sample_t)
    
    img = cv2.imread("demo/demo.png")
    img = cv2.resize(img,(1024,512))
    x = 0
    y= 200
    # img = cv2.imread("demo/demo2.png")
    # img = cv2.resize(img,(1280,760))
    # x = 0
    # y= 200
    crop_img = img[x:x+512,y:y+512]
    cv2.imwrite('demo/demo_crop.png',crop_img)
    
    crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(crop_img_rgb).float().to("cuda").unsqueeze(0)
    img= img.permute(0, 3, 1, 2)
 
    
    for i in range(3):
        img[:,i,:,:] =img[:,i,:,:]-127.5
        img[:,i,:,:] =img[:,i,:,:]/127.5
    
   
    
    t = torch.full((1,),0, dtype = torch.long).cuda()
        # zeros = torch.zeros((batch_size,),requires_grad=False).cuda().long()
    # noise_gt= default(None, lambda: torch.randn([img.shape[0],img.shape[1],int(img.shape[2]/4),int(img.shape[3]/4)],requires_grad=False)).cuda()
    # noise_gt2= default(None, lambda: torch.randn([img.shape[0],img.shape[1],int(img.shape[2]/4),int(img.shape[3]/4)],requires_grad=False)).cuda()
    noise_gt= default(None, lambda: torch.randn(img.shape,requires_grad=False)).cuda()
    noise = noise_gt
    noise_gt2= default(None, lambda: torch.randn(img.shape,requires_grad=False)).cuda()
    # noise = torch.nn.functional.interpolate(noise_gt, scale_factor=4, mode='bilinear', align_corners=False).cuda()
    alpha_cumprod = torch.cumprod(1.-sigmoid_beta_schedule(sample_t ), dim=0).cuda()
    
    img_sample = extract(torch.sqrt( alpha_cumprod), t, img.shape) * img +extract(torch.sqrt(1- alpha_cumprod), t, img.shape) * noise   
    
    noise_pred = model.encode_decode([img_sample.float(),t],None)
    noise_pred_up = torch.nn.functional.interpolate(noise_pred[1], scale_factor=4, mode='bilinear', align_corners=False).cuda()
    img_start =  extract(torch.sqrt(1. / alpha_cumprod), t, img_sample.shape) * img_sample -extract(torch.sqrt(1. / alpha_cumprod - 1), t, img_sample.shape) * noise_pred[1]
    
    # img =  torch.nn.functional.interpolate(img, scale_factor=0.25, mode='bilinear', align_corners=False).cuda()
    # img_sample2 = extract(torch.sqrt( alpha_cumprod), t, img.shape) * img +extract(torch.sqrt(1- alpha_cumprod), t, img.shape) * noise_gt
    # img_start =  extract(torch.sqrt(1. / alpha_cumprod), t, img_sample2.shape) * img_sample2 -extract(torch.sqrt(1. / alpha_cumprod - 1), t, img_sample.shape) * noise_pred[1]
    
    
    npimg = img_sample[0].cpu()
    npimg = (npimg+1)/2
    npimg = torch.clamp(npimg , 0, 1)
    npimg = T(npimg)
    npimg.save('demo/demo_s.jpg')
    npimg = img_start[0].cpu()
    npimg = (npimg+1)/2
    npimg = torch.clamp(npimg,0,1)
    print(torch.max(img_sample),torch.max(noise_gt),torch.max(noise_pred_up),torch.min(img_start))
    npimg = T(npimg)
    npimg.save('demo/demo_d.jpg')

    print(torch.mean(torch.square(noise_pred[1]-noise_gt)))
    print(torch.mean(torch.square(noise_gt-noise_gt2)))
  
if __name__ == '__main__':
    
    main()
