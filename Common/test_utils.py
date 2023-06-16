import numpy as np
import random
import torch
import open3d as o3d
import os
import glob

def seed_reset(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def pc_normalize(pc,return_len=False):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    if return_len:
        return m
    # print('not normalized')
    return pc

def sphere_generator(opts, static=True):
    ball = np.loadtxt('template/balls/%d.xyz'% opts.np)[:, :3]
    ball = pc_normalize(ball)
    if static:
        ball = np.expand_dims(ball, axis=0)
        ball = np.tile(ball, (opts.bs, 1,1))
        ball = torch.Tensor(ball).cuda()
    else:
        ball_temp = np.zeros((opts.bs, opts.np, 3))
        for i in range(opts.bs):
            idx = np.random.choice(ball.shape[0], opts.np)
            ball_temp[i] = ball[idx]
        ball = torch.Tensor(ball_temp).cuda()
        
    return ball
def noise_generator(opts, masks=None):
    if masks is None:
        if opts.n_rand:
            print("All random latent per point")
            noise = np.random.normal(0, opts.nv, (opts.bs, opts.np, opts.nz))
        else:
            noise = np.random.normal(0, opts.nv, (opts.bs, 1, opts.nz))
            noise = np.tile(noise, (1, opts.np, 1))

        if opts.n_mix:
            noise2 = np.random.normal(0, opts.nv, (opts.bs, opts.nz))
            for i in range(opts.bs):
                idx = np.arange(opts.np)
                np.random.shuffle(idx)
                num=int(random.random() * opts.np)
                noise[i, idx[:num]] = noise2[i]
    else:
        pass
    return torch.Tensor(noise).cuda()
    
def save_single_pointcloud(pc, opts, out_folder, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    name = '%s_%s.pcd' % (str(opts.choice).lower(), str(len(glob.glob(os.path.join(out_folder, '*.pcd')))).zfill(4))
    filename = os.path.join(*[out_folder, name])
    o3d.io.write_point_cloud(filename, pcd)

def save_pointclouds(pc, opts, color=None):
    cat = str(opts.choice).lower()
    out_folder = os.path.join(*[opts.exp_name, cat])
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for p in pc:
        save_single_pointcloud(p, opts, out_folder, color)

def get_color(sphere):
    color = (1+sphere)/2
    if type(color) == torch.Tensor:
        color = color.cpu().detach().numpy()
    return color