import os
import pprint
# pp = pprint.PrettyPrinter()
# from datetime import datetime

from Generation.config import opts
from Generation.Generator import Generator
from Generation.Encoder import Encoder, Remapper
from Generation.CompLoader import CompLoader

# from torch.autograd import Variable
# import random
import numpy as np
import torch

# from Common.test_utils import *

# from fps.fps_v1 import FPS
from tqdm import tqdm
import open3d as o3d
import matplotlib.pylab  as plt

# from Common.point_operation import normalize_point_cloud
# from Common.visu_utils import plot_pcd_multi_rows
import sys
sys.path.append(os.path.join(os.getcwd(),"metrics"))
from metrics.Chamfer3D.dist_chamfer_3D import chamfer_3DDist as chamfer_dist
import pandas as pd

import glob

def load_weights(path, epoch_mode="200_ae"):
    g_path = os.path.join(path, epoch_mode+'_G.pth')
    e_path = os.path.join(path, epoch_mode+'_E.pth')
    m_path = os.path.join(path, epoch_mode+'_M.pth')
    mapper = Remapper().cuda().eval()
    generator = Generator(opts).cuda().eval()
    encoder = Encoder().cuda().eval()
    checkpoint_g = torch.load(g_path)
    checkpoint_e = torch.load(e_path)
    checkpoint_m = torch.load(m_path)
    
    generator.load_state_dict(checkpoint_g['G_model'])
    encoder.load_state_dict(checkpoint_e['E_model'])
    mapper.load_state_dict(checkpoint_m['remap_model'])
    return generator, encoder, mapper

def plot_pcd_multi_rows(filename, pcds, titles, suptitle='', sizes=None, cmap='Greys', zdir='y',
                         xlim=(-0.4, 0.4), ylim=(-0.4, 0.4), zlim=(-0.4, 0.4)):
    if sizes is None:
        sizes = [0.2 for i in range(len(pcds[0]))]

    #print(len(pcds),len(pcds[0]))
    fig = plt.figure(figsize=(len(pcds[0]) * 3, len(pcds)*3)) # W,H
    for i in range(len(pcds)):
        elev = 30
        azim = -45
        for j, (pcd, size) in enumerate(zip(pcds[i], sizes)):
            color = np.zeros(pcd.shape[0])
            ax = fig.add_subplot(len(pcds), len(pcds[i]), i * len(pcds[i]) + j + 1, projection='3d')
            #print(len(pcds), len(pcds[i]), i * len(pcds[i]) + j + 1)
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[i][j])
            #ax.text(0, 0, titles[i][j], color="green")
            ax.set_axis_off()
            #ax.set_xlabel(titles[i][j])

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

    #plt.xticks(np.arange(len(pcds)), titles[:len(pcds)])

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def tensor_to_pcd(tensor, name):
    t = tensor.squeeze()
    try:
        t = t.cpu().detach().numpy()
    except:
        try:
            t = t.numpy()
        except:
            pass
    if t.shape[0] == 3:
        t = t.transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(t)
    o3d.io.write_point_cloud(name, pcd)

def pc_normalize(pc,return_len=False):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    if return_len:
        return m
    # print('not normalized')
    return pc

def test_ae(data_path, weight_path, save_path, target_epoch, cat='airplane'):
    sphere_2048 = pc_normalize(np.loadtxt('template/balls/2048.xyz')[:,:3])
    sphere_2048 = torch.Tensor(np.expand_dims(sphere_2048, axis=0)).cuda()
    generator, encoder, mapper = load_weights(weight_path, str(target_epoch)+"_ae")
    test_dataset = CompLoader(data_path, cat, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    target_path = os.path.join(save_path, cat)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for idx, (gt, partial, category) in tqdm(enumerate(test_loader, 0),total=len(test_loader)):
        gt = gt.cuda()
        latent = encoder(gt).unsqueeze(1)
        latent = torch.tile(latent,(1,opts.np, 1))
        # print(latent.shape, sphere_2048.shape)
        recon = generator(sphere_2048, latent)

        gt_name = os.path.join(target_path, str(idx)+'_ae_gt.pcd')
        partial_name = os.path.join(target_path, str(idx)+'_ae_partial.pcd')
        recon_name = os.path.join(target_path, str(idx)+'_ae_recon.pcd')

        tensor_to_pcd(gt, gt_name)
        tensor_to_pcd(partial, partial_name)
        tensor_to_pcd(recon, recon_name)

def test_completion(data_path, weight_path, save_path, target_epoch, cat='airplane'):
    sphere_2048 = pc_normalize(np.loadtxt('template/balls/2048.xyz')[:,:3])
    sphere_2048 = torch.Tensor(np.expand_dims(sphere_2048, axis=0)).cuda()
    generator, encoder, mapper = load_weights(weight_path, str(target_epoch)+"_lgan")
    test_dataset = CompLoader(data_path, cat, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    target_path = os.path.join(save_path, cat)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for idx, (gt, partial, category) in tqdm(enumerate(test_loader, 0),total=len(test_loader)):
        partial = partial.cuda()
        latent = encoder(partial)
        latentw = mapper(latent).unsqueeze(1)
        latentw = torch.tile(latentw,(1,opts.np, 1))
        # print(latent.shape, sphere_2048.shape)
        recon = generator(sphere_2048, latentw)

        gt_name = os.path.join(target_path, str(idx)+'_completion_gt.pcd')
        partial_name = os.path.join(target_path, str(idx)+'_completion_partial.pcd')
        recon_name = os.path.join(target_path, str(idx)+'_completion_recon.pcd')

        tensor_to_pcd(gt, gt_name)
        tensor_to_pcd(partial, partial_name)
        tensor_to_pcd(recon, recon_name)

def mass_test(data_path, log_path):
    weight_path = log_path
    save_path = os.path.join(weight_path, "results")
    categories = ['airplane', 'car', 'chair', 'guitar', 'table']
    for c in categories:
        test_ae(data_path, weight_path, save_path, 100, c)
    for c in categories:
        test_completion(data_path, weight_path, save_path, 100, c)

def sphere_generator(bs=1, n_p = 2048):
    ball = np.loadtxt('template/balls/%d.xyz'% n_p)[:, :3]
    ball = pc_normalize(ball)
    ball = ball[np.newaxis, :, :]
    # ball = np.expand_dims(ball, axis=0)
    ball = np.tile(ball, (bs, 1,1))
    ball = torch.Tensor(ball).cuda()
    return ball

def get_samples(path='dataset', cat = 'airplane', num_exp=4):
    # cats = ['airplane', 'car', 'chair', 'guitar', 'table']
    folder = os.path.join(path, cat)
    object_list = sorted(glob.glob(os.path.join(folder, "*/models")))[180:]
    gtlist = []
    plist = []
    for obj in object_list:
        gt = np.load(os.path.join(obj, 'gt_pc.npy')).astype(np.float32).T
        partial_list = sorted(glob.glob(os.path.join(obj, 'part*')))
        partial_list = partial_list[:num_exp]
        plist_temp = []
        for p in partial_list:
            plist_temp.append(np.load(os.path.join(p)).astype(np.float32).T)
        plist.append(torch.Tensor(np.array(plist_temp)))
        gtlist.append(gt)

    return gtlist, plist

def draw_diverse_sample(data_path, weight_path, save_path, target_epoch, num_exp=4, cat='airplane'):
    # weight_path, log_dir, num_rows = 10
    print("Starting sampling conditional SP-GAN")
    generator, encoder, mapper = load_weights(weight_path, str(target_epoch)+"_lgan")
    pcd_dir = os.path.join(save_path, cat+"_pcd")
    if not os.path.exists(pcd_dir):
        os.makedirs(pcd_dir)
    
    sphere_2048 = sphere_generator(bs=num_exp, n_p = 2048)
    # fix_zc = noise_generator_cond(bs=args.num_rows)
    gtlist, plist = get_samples(data_path, cat, num_exp)
    print("Start sampling")
    pcds_list = []
    title_list = []

    cdlist = []
    cdloss = chamfer_dist()
    for i in range(len(gtlist)):
        # title = ["S_%d" % (i * grid_y + j) for j in range(grid_y)]
        # title = ["airplane_%d" % i, "car_%d" % i, "chair_%d" % i, "guitar_%d" % i, "table_%d" % i]
        title = ["GT_%d" %i]
        with torch.no_grad():
            partial = plist[i].cuda()
            latent = encoder(partial)
            latentw = mapper(latent).unsqueeze(1)
            latentw = torch.tile(latentw,(1,opts.np, 1))
            recon = generator(sphere_2048, latentw)
            recon = recon.transpose(2, 1)
            partial = partial.transpose(2, 1)
            recon = recon.cpu().detach().numpy()
            # recon = normalize_point_cloud(recon)
            # recon = 0.75* recon
            partial = partial.cpu().detach().numpy()
            # partial = normalize_point_cloud(partial)
            # partial = 0.75 * partial
        sample_pcs = [gtlist[i].T]
        cdlist_one = []
        for j in range(num_exp):
            # print(gtlist[i].shape, recon[j].shape)
            dist1_r, dist2_r, idx1_r, idx2_r  = cdloss(torch.Tensor((gtlist[i].T)[np.newaxis,:,:]).cuda() , torch.Tensor((recon[j])[np.newaxis,:,:]).cuda())
            # print(dist1_r.mean().item(), dist2_r.mean().item())
            cd = (dist1_r.mean().item()+dist2_r.mean().item()) / 2
            sample_pcs.append(partial[j])
            sample_pcs.append(recon[j])
            title.append("Partial_%d_%d" % (i, j))
            title.append("Completion_%d_%d_" % (i, j)+format(cd, ".6f"))
            cdlist_one.append(cd)

        pcds_list.append(sample_pcs)
        title_list.append(title)
        cdlist.append(cdlist_one)

        for i, t in enumerate(title):
            tensor_to_pcd(sample_pcs[i], os.path.join(pcd_dir, t+".pcd"))

    plot_name = os.path.join(save_path, cat+"_results.png")

    plot_pcd_multi_rows(plot_name, pcds_list, title_list)
    print("Sampled point clouds has been plotted in table at:", save_path)
    small_part_path = os.path.join(save_path, cat+"_partimg")
    if not os.path.exists(small_part_path):
        os.makedirs(small_part_path)
    print("Start making parts of result table")
    for i in range(len(pcds_list)):
        for j in range(len(pcds_list[i])):
            small_pcds_list = [[pcds_list[i][j]]]
            small_title_list = [[title_list[i][j]]]
            small_plot_name = os.path.join(small_part_path, title_list[i][j]+'.png')
            plot_pcd_multi_rows(small_plot_name, small_pcds_list, small_title_list)
    print("Parts of result table saved")
    return np.array(cdlist)


if __name__ == "__main__":
    data_path = "dataset"
    save_target = "divplot"
    air_cdtable = draw_diverse_sample(data_path=data_path, weight_path='latest_weights', target_epoch=100, save_path=save_target, num_exp=16, cat='airplane')
    print(air_cdtable.shape)
    car_cdtable = draw_diverse_sample(data_path=data_path, weight_path='latest_weights', target_epoch=100, save_path=save_target, num_exp=16,cat='car')
    chair_cdtable = draw_diverse_sample(data_path=data_path, weight_path='latest_weights', target_epoch=100, save_path=save_target, num_exp=16, cat='chair')
    guitar_cdtable = draw_diverse_sample(data_path=data_path, weight_path='latest_weights', target_epoch=100, save_path=save_target, num_exp=16, cat='guitar')
    table_cdtable = draw_diverse_sample(data_path=data_path, weight_path='latest_weights', target_epoch=100, save_path=save_target, num_exp=16,cat='table')
    total_cd_table = np.concatenate([air_cdtable, car_cdtable, chair_cdtable, guitar_cdtable, table_cdtable], axis=0)
    print(total_cd_table.shape)
    cd_table = pd.DataFrame(total_cd_table)
    cd_table.to_excel(os.path.join(save_target, "cd_table.xlsx"))