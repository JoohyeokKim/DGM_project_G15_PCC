import numpy as np
import random
from Generation.config import opts
from Generation.Generator import Generator
from Common.test_utils import *

seed = 1990
seed_reset(seed)
from Common.test_utils import *

model = Generator(opts=opts)
model.cuda()
model.eval()
load_weights(model, opts)
seed_reset(seed)
# z = noise_generator(opts)[:,:,:]
sphere_2048 = np.loadtxt('template/balls/2048.xyz')[:,:3]
sphere_2048 = torch.Tensor(np.expand_dims(sphere_2048, axis=0)).cuda()
noise_template = np.random.normal(0, opts.nv, (opts.bs, 1, opts.nz))
z_1024 = torch.Tensor(np.tile(noise_template, (1, 1024, 1))).cuda()
z_2048 = torch.Tensor(np.tile(noise_template, (1, 2048, 1))).cuda()
z_4096 = torch.Tensor(np.tile(noise_template, (1, 4096, 1))).cuda()

