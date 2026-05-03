import torch
import numpy as np
from matplotlib import pyplot as plt
from mydataset import Trainer,Unet, GaussianDiffusion,default,extract,sigmoid_beta_schedule
from accelerate import Accelerator
from torchvision import transforms
import cv2
import PIL
from collections import OrderedDict
from tqdm.auto import tqdm

device = 'cuda'
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    style = True,
).to(device)


data = torch.load('./results/model-1.pt', map_location=device)

new_dict = OrderedDict()  
x = data['model']
for key in x:
    if 'model.'in key:
        new_dict[key[6:]]=x[key]
model.load_state_dict(new_dict)


diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1',            # L1 or L2
    min_snr_gamma= True,
).to(device)


# sampled_images = diffusion.sample(batch_size = 1)




from PIL import Image  


img = cv2.imread("../data/gta/images/00333.png")
img = cv2.resize(img,(1280,760))
x = 200
y= 200
img = cv2.imread("../data/cityscapes/leftImg8bit/train/darmstadt/darmstadt_000006_000019_leftImg8bit.png")
img = cv2.resize(img,(1024,512))
x = 0
y= 200

resized_img = cv2.resize(img[x:x+512,y:y+512], (128,128), interpolation=cv2.INTER_LINEAR)#
cv2.imwrite('demo.png',resized_img)

resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
imgout = torch.from_numpy(resized_img_rgb).to("cuda").div(255.0).unsqueeze(0)
imgout2 = imgout.permute(0, 3, 1, 2)

imgout2=imgout2 * 2 - 1
noise = default(None, lambda: torch.randn_like(imgout2))


t = torch.full((1,),750, device = device, dtype = torch.long)
q_sample = extract(torch.sqrt(torch.cumprod(1.-sigmoid_beta_schedule(1000), dim=0)).cuda(), t, imgout2.shape) * imgout2 +extract(torch.sqrt(1-torch.cumprod(1.-sigmoid_beta_schedule(1000).cuda(), dim=0)), t, imgout2.shape) * noise
output = model(q_sample.float(),t,2)


# alphas_cumprod = torch.cumprod(1.-sigmoid_beta_schedule(100), dim=0)
# snr = alphas_cumprod / (1 - alphas_cumprod)
# print(snr[0],snr[10],snr[20],snr[30])
# snr_cl = torch.clamp(snr,max=5)
# snr_cl2 = torch.clamp(snr,min=1)
# print(alphas_cumprod)
# print(snr_cl/snr)
# print(snr/snr_cl2)
# print(torch.mean(torch.square(output -noise)))
print(torch.sqrt(torch.cumprod(1.-sigmoid_beta_schedule(100), dim=0)))
# print(torch.cumprod(1.-sigmoid_beta_schedule(1000), dim=0)[500])
x_start = diffusion.predict_start_from_noise(q_sample, t, output )

T = transforms.ToPILImage()
print(x_start.shape,torch.max(x_start))
npimg = x_start[0].cpu()
npimg = (npimg+1)/2
npimg = torch.clamp(npimg , 0, 1)
npimg = T(npimg)
npimg.save('./demo_d.jpg')

print(q_sample.shape)
q_sample_ =(q_sample+1)/2
q_sample_ = torch.clamp(q_sample_ , 0, 1)
npimg = q_sample_ [0].cpu()
npimg = T(npimg)
npimg.save('./demo_q.jpg')

print(noise.shape)
npimg =noise [0].cpu()
npimg = T(npimg)
npimg.save('./demo_n.jpg')



import torch.nn.functional as F
sample_t = 1000
img = q_sample#torch.randn([1,3,128,128], device = 'cuda')
betas = sigmoid_beta_schedule(sample_t).cuda()
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).cuda()
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_log_variance_clipped= torch.log(posterior_variance.clamp(min =1e-20))
for t in tqdm(reversed(range(0, sample_t-250)), desc = 'sampling loop time step', total = sample_t):
    batched_times = torch.full((1,), t, device = 'cuda', dtype = torch.long)
    with torch.no_grad(): 
        output = model(img.float(),batched_times,2)
    x_start = diffusion.predict_start_from_noise(img,batched_times, output )
    
    x_start.clamp_(-1., 1.)
    model_mean= (
        extract(betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), batched_times, img.shape) * x_start +
        extract( (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod), batched_times, img.shape) * img
    )
    posterior_log_variance = extract(posterior_log_variance_clipped, batched_times, img.shape)
    noise = torch.randn([1,3,128,128], device = 'cuda') if t > 0 else 0. # no noise if t == 0
    img = model_mean + (0.5 * posterior_log_variance).exp() * noise
   
    
npimg = img[0].cpu()
print(torch.max(npimg),torch.min(npimg))
npimg = (npimg+1)/2
npimg = T(npimg)
npimg.save('./demo_sample.jpg')