import os
import pprint
pp = pprint.PrettyPrinter()
from datetime import datetime

from Generation.config import opts
from Generation.Generator import Generator
import random
import numpy as np
import torch

from Common.test_utils import *

from fps.fps_v1 import FPS

def massive_generate(model, opts, gen_num, color=True):
    # if gen_num % opts.bs == 0:
    #     iter = gen_num // opts.bs
    # else:
    #     iter = gen_num // opts.bs + 1 # if gennum = 32 bs = 8 32 // 8 = 4
    sphere = sphere_generator(opts)
    iteration = gen_num // opts.bs
    if color:
        s_color = get_color(sphere[0])
    else:
        s_color = None
    for i in range(iteration):    
        z = noise_generator(opts)
        out_pc = model(sphere, z)
        out_pc = out_pc.transpose(2,1)
        out_pc = out_pc.cpu().detach().numpy()
        save_pointclouds(out_pc, opts, color=s_color)
    
    cut = gen_num%opts.bs
    if cut != 0:
        z = noise_generator(opts)
        out_pc = model(sphere[:cut], z[:cut])
        out_pc = out_pc.transpose(2,1)
        out_pc = out_pc.cpu().detach().numpy()
        save_pointclouds(out_pc, opts, color=s_color)


def test_latent_pn(opts, seed):
    shift_choice(opts, 'table')
    opts.np = 1024
    model = Generator(opts=opts)
    model.cuda()
    model.eval()
    load_weights(model, opts)
    seed_reset(seed)
    zt1 = noise_generator(opts)[:,0,:]

    opts.choice = 'table'
    opts.np = 2048
    model = Generator(opts=opts)
    model.cuda()
    model.eval()
    load_weights(model, opts)
    seed_reset(seed)
    zt2 = noise_generator(opts)[:,0,:]

    opts.choice = 'table'
    opts.np = 4096
    model = Generator(opts=opts)
    model.cuda()
    model.eval()
    load_weights(model, opts)
    seed_reset(seed)
    zt3 = noise_generator(opts)[:,0,:]

    shift_choice(opts, 'human')
    opts.np = 1024
    model = Generator(opts=opts)
    model.cuda()
    model.eval()
    load_weights(model, opts)
    seed_reset(seed)
    zh1 = noise_generator(opts)[:,0,:]

    opts.choice = 'human'
    opts.np = 2048
    model = Generator(opts=opts)
    model.cuda()
    model.eval()
    load_weights(model, opts)
    seed_reset(seed)
    zh2 = noise_generator(opts)[:,0,:]

    opts.choice = 'human'
    opts.np = 4096
    model = Generator(opts=opts)
    model.cuda()
    model.eval()
    load_weights(model, opts)
    seed_reset(seed)
    zh3 = noise_generator(opts)[:,0,:]

    opts.np = 1024
    shift_choice(opts, 'Chair')
    model = Generator(opts=opts)
    model.cuda()
    model.eval()
    load_weights(model, opts)
    seed_reset(seed)
    zc1 = noise_generator(opts)[:,0,:]

    opts.np = 2048
    opts.choice = 'Chair'
    model = Generator(opts=opts)
    model.cuda()
    model.eval()
    load_weights(model, opts)
    seed_reset(seed)
    zc2 = noise_generator(opts)[:,0,:]

    opts.np = 4096
    opts.choice = 'Chair'
    model = Generator(opts=opts)
    model.cuda()
    model.eval()
    load_weights(model, opts)
    seed_reset(seed)
    zc3 = noise_generator(opts)[:,0,:]
    
    mse = torch.nn.MSELoss()
    zlist = [zh1, zh2, zh3, zt1, zt2, zt3, zc1, zc2, zc3]

    for i in zlist:
        for j in zlist:
            print(mse(i,j))
    

def test_pn_gen(opts, seed):
    folder_name = 'notnorm'
    opts.bs = 1
    model = Generator(opts=opts)
    model.cuda()
    model.eval()
    # sphere_1024 = pc_normalize(np.loadtxt('template/balls/1024.xyz')[:,:3])
    # sphere_1024 = torch.Tensor(np.expand_dims(sphere_1024, axis=0)).cuda()
    sphere_2048 = pc_normalize(np.loadtxt('template/balls/2048.xyz')[:,:3])
    sphere_2048 = torch.Tensor(np.expand_dims(sphere_2048, axis=0)).cuda()
    # sphere_1792 = pc_normalize(np.loadtxt('template/balls/4096.xyz')[:,:3])
    # sphere_1792 = torch.Tensor(np.expand_dims(sphere_1792, axis=0)).cuda()
    sphere_2048_ = pc_normalize(np.loadtxt('template/balls/2048.xyz')[:,:3])
    fps_1024 = FPS(sphere_2048_, 1024)
    sphere_1024 = fps_1024.fit().get()
    sphere_1024 = torch.Tensor(np.expand_dims(sphere_1024, axis=0)).cuda()
    fps_1536 = FPS(sphere_2048_, 1536)
    sphere_1536 = fps_1536.fit().get()
    sphere_1536 = torch.Tensor(np.expand_dims(sphere_1536, axis=0)).cuda()
    fps_1792 = FPS(sphere_2048_, 1792)
    sphere_1792 = fps_1792.fit().get()
    sphere_1792 = torch.Tensor(np.expand_dims(sphere_1792, axis=0)).cuda()
    sphere_1792_2 = pc_normalize(np.loadtxt('template/balls/4096.xyz')[:,:3])
    fps_2048 = FPS(sphere_1792_2, 2048)
    sphere_2048_2 = fps_2048.fit().get()
    sphere_2048_2 = torch.Tensor(np.expand_dims(sphere_2048_2, axis=0)).cuda()

    seed_reset(seed)
    noise_template = np.random.normal(0, opts.nv, (opts.bs, 1, opts.nz))
    z_1024 = torch.Tensor(np.tile(noise_template, (1, 1024, 1))).cuda()
    z_2048 = torch.Tensor(np.tile(noise_template, (1, 2048, 1))).cuda()
    z_1792 = torch.Tensor(np.tile(noise_template, (1, 1792, 1))).cuda()
    z_1536 = torch.Tensor(np.tile(noise_template, (1, 1536, 1))).cuda()
    z_2048_2 = torch.Tensor(np.tile(noise_template, (1, 2048, 1))).cuda()

    shift_choice(opts, 'Chair')
    load_weights(model, opts)
    out_1024 = model(sphere_1024, z_1024)
    out_1024 = out_1024.transpose(2,1)
    out_1024 = out_1024.cpu().detach().numpy()[0]

    out_2048 = model(sphere_2048, z_2048)
    out_2048 = out_2048.transpose(2,1)
    out_2048 = out_2048.cpu().detach().numpy()[0]

    out_1792 = model(sphere_1792, z_1792)
    out_1792 = out_1792.transpose(2,1)
    out_1792 = out_1792.cpu().detach().numpy()[0]

    out_1536 = model(sphere_1536, z_1536)
    out_1536 = out_1536.transpose(2,1)
    out_1536 = out_1536.cpu().detach().numpy()[0]

    out_2048_2 = model(sphere_2048_2, z_2048_2)
    out_2048_2 = out_2048_2.transpose(2,1)
    out_2048_2 = out_2048_2.cpu().detach().numpy()[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1024)
    filename = folder_name + '/chair_1024_2.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_2048)
    filename = folder_name + '/chair_2048.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1792)
    filename = folder_name + '/chair_1792.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1536)
    filename = folder_name + '/chair_1536.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_2048_2)
    filename = folder_name + '/chair_2048_2.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    shift_choice(opts, 'table')
    load_weights(model, opts)
    out_1024 = model(sphere_1024, z_1024)
    out_1024 = out_1024.transpose(2,1)
    out_1024 = out_1024.cpu().detach().numpy()[0]

    out_2048 = model(sphere_2048, z_2048)
    out_2048 = out_2048.transpose(2,1)
    out_2048 = out_2048.cpu().detach().numpy()[0]

    out_1792 = model(sphere_1792, z_1792)
    out_1792 = out_1792.transpose(2,1)
    out_1792 = out_1792.cpu().detach().numpy()[0]

    out_1536 = model(sphere_1536, z_1536)
    out_1536 = out_1536.transpose(2,1)
    out_1536 = out_1536.cpu().detach().numpy()[0]

    out_2048_2 = model(sphere_2048_2, z_2048_2)
    out_2048_2 = out_2048_2.transpose(2,1)
    out_2048_2 = out_2048_2.cpu().detach().numpy()[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1024)
    filename = folder_name + '/table_1024_2.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_2048)
    filename = folder_name + '/table_2048.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1792)
    filename = folder_name + '/table_1792.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1536)
    filename = folder_name + '/table_1536.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_2048_2)
    filename = folder_name + '/table_2048_2.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    shift_choice(opts, 'human')
    load_weights(model, opts)
    out_1024 = model(sphere_1024, z_1024)
    out_1024 = out_1024.transpose(2,1)
    out_1024 = out_1024.cpu().detach().numpy()[0]

    out_2048 = model(sphere_2048, z_2048)
    out_2048 = out_2048.transpose(2,1)
    out_2048 = out_2048.cpu().detach().numpy()[0]

    out_1792 = model(sphere_1792, z_1792)
    out_1792 = out_1792.transpose(2,1)
    out_1792 = out_1792.cpu().detach().numpy()[0]

    out_1536 = model(sphere_1536, z_1536)
    out_1536 = out_1536.transpose(2,1)
    out_1536 = out_1536.cpu().detach().numpy()[0]

    out_2048_2 = model(sphere_2048_2, z_2048_2)
    out_2048_2 = out_2048_2.transpose(2,1)
    out_2048_2 = out_2048_2.cpu().detach().numpy()[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1024)
    filename = folder_name + '/human_1024_2.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_2048)
    filename = folder_name + '/human_2048.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1792)
    filename = folder_name + '/human_1792.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1536)
    filename = folder_name + '/human_1536.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_2048_2)
    filename = folder_name + '/human_2048_2.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    shift_choice(opts, 'airplane')
    load_weights(model, opts)
    out_1024 = model(sphere_1024, z_1024)
    out_1024 = out_1024.transpose(2,1)
    out_1024 = out_1024.cpu().detach().numpy()[0]

    out_2048 = model(sphere_2048, z_2048)
    out_2048 = out_2048.transpose(2,1)
    out_2048 = out_2048.cpu().detach().numpy()[0]

    out_1792 = model(sphere_1792, z_1792)
    out_1792 = out_1792.transpose(2,1)
    out_1792 = out_1792.cpu().detach().numpy()[0]

    out_1536 = model(sphere_1536, z_1536)
    out_1536 = out_1536.transpose(2,1)
    out_1536 = out_1536.cpu().detach().numpy()[0]

    out_2048_2 = model(sphere_2048_2, z_2048_2)
    out_2048_2 = out_2048_2.transpose(2,1)
    out_2048_2 = out_2048_2.cpu().detach().numpy()[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1024)
    filename = folder_name + '/airplane_1024_2.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_2048)
    filename = folder_name + '/airplane_2048.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1792)
    filename = folder_name + '/airplane_1792.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_1536)
    filename = folder_name + '/airplane_1536.pcd'
    o3d.io.write_point_cloud(filename, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_2048_2)
    filename = folder_name + '/airplane_2048_2.pcd'
    o3d.io.write_point_cloud(filename, pcd)






if __name__ == '__main__':
    test_latent_pn(opts, 1990)
    exit()
    
    model = Generator(opts=opts)
    model.cuda()
    model.eval()

    load_weights(model, opts)
    seed = 1990
    seed_reset(seed)

    # sphere = sphere_generator(opts)[0]
    # color = get_color(sphere)



    massive_generate(model, opts, 50, True)


    # for s in sp:
    #     c_color = get_color(s)
    #     c_color = c_color#.cpu().detach().numpy()
    #     save_pointclouds(sp, opts, color=c_color)
