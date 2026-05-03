import shutup
shutup.please()

from mydataset import Trainer,Unet, GaussianDiffusion
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    style=True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1',            # L1 or L2
    min_snr_gamma= True
)
trainer = Trainer(
    diffusion,
    ['../data/gta/images',
    '../data/cityscapes/leftImg8bit/train'],
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()