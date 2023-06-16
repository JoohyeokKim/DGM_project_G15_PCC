import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from numpy.linalg import norm
import sys,os
import torch.nn.functional as F
#from Common.Const import GPU
from torch.autograd import Variable#, grad
sys.path.append(os.path.join(os.getcwd(),"metrics"))
# from metrics.CD_EMD.cd.chamferdist.ChamferDistance import chamferFunction
# Import CUDA version of approximate EMD, from https://github.com/zekunhao1995/pcgan-pytorch/
# from StructuralLosses.match_cost import match_cost
# from StructuralLosses.nn_distance import nn_distance
from torch.autograd import Variable
from Common.modules import pairwise_dist
from torch.distributions import Beta
import functools



def dist_o2l(p1, p2):
    # distance from origin to the line defined by (p1, p2)
    p12 = p2 - p1
    u12 = p12 / np.linalg.norm(p12)
    l_pp = np.dot(-p1, u12)
    pp = l_pp*u12 + p1
    return np.linalg.norm(pp)

def para_count(models):
    count = 0
    for model in models:
        count +=  sum(param.numel() for param in model.parameters())
    return count

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# # Import CUDA version of CD, borrowed from https://github.com/ThibaultGROUEIX/AtlasNet
# try:
#     from . chamfer_distance_ext.dist_chamfer import chamferDist
#     CD = chamferDist()
#     def distChamferCUDA(x,y):
#         return CD(x,y,gpu)
# except:

class CrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=True):
        super(CrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, preds, gts):
        gts = gts.contiguous().view(-1)

        if self.smoothing:
            eps = 0.2
            n_class = preds.size(1)

            one_hot = torch.zeros_like(preds).scatter(1, gts.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(preds, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(preds, gts, reduction='mean')

        return loss

class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self,preds,gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2


    def batch_pairwise_dist(self,x,y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        #xx = torch.bmm(x, x.transpose(2,1))
        xx = torch.sum(x ** 2, dim=2, keepdim=True)
        yy = torch.sum(y ** 2, dim=2, keepdim=True)
        xy = -2 * torch.bmm(x, y.permute(0, 2, 1))
        dist = xy + xx + yy.permute(0, 2, 1)  # [B, N, N]
        return dist
        yy = torch.bmm(y, y.transpose(2,1))
        zz = torch.bmm(x, y.transpose(2,1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        #brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2,1) + ry - 2*zz)

        return P

# def batch_pairwise_dist(self,x,y):
#
#    bs, num_points_x, points_dim = x.size()
#     _, num_points_y, _ = y.size()
#
#     xx = torch.sum(x ** 2, dim=2, keepdim=True)
#     yy = torch.sum(y ** 2, dim=2, keepdim=True)
#     yy = yy.permute(0, 2, 1)
#
#     xi = -2 * torch.bmm(x, y.permute(0, 2, 1))
#     dist = xi + xx + yy  # [B, N, N]
#     return dist


def dist_simple(x,y,loss="l2"):
    if loss == "l2":
        dist = torch.sum((x - y) ** 2, dim=-1).sum(dim=1).float()
    else:
        dist = torch.sum(torch.abs(x - y), dim=-1).sum(dim=1).float()

    return dist.mean()


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }



def get_voxel_occ_dist(all_clouds, clouds_flag='gen', res=28, bound=0.5, bs=128, warning=True):
    if np.any(np.fabs(all_clouds) > bound) and warning:
        print('{} clouds out of cube bounds: [-{}; {}]'.format(clouds_flag, bound, bound))

    n_nans = np.isnan(all_clouds).sum()
    if n_nans > 0:
        print('{} NaN values in point cloud tensors.'.format(n_nans))

    p2v_dist = np.zeros((res, res, res), dtype=np.uint64)

    step = 1. / res
    v_bs = -0.5 + np.arange(res + 1) * step

    nbs = all_clouds.shape[0] // bs + 1
    for i in range(nbs):
        clouds = all_clouds[bs * i:bs * (i + 1)]

        preiis = clouds[:, :, 0].reshape(1, -1)
        preiis = np.logical_and(v_bs[:28].reshape(-1, 1) <= preiis, preiis < v_bs[1:].reshape(-1, 1))
        iis = preiis.argmax(0)
        iis_values = preiis.sum(0) > 0

        prejjs = clouds[:, :, 1].reshape(1, -1)
        prejjs = np.logical_and(v_bs[:28].reshape(-1, 1) <= prejjs, prejjs < v_bs[1:].reshape(-1, 1))
        jjs = prejjs.argmax(0)
        jjs_values = prejjs.sum(0) > 0

        prekks = clouds[:, :, 2].reshape(1, -1)
        prekks = np.logical_and(v_bs[:28].reshape(-1, 1) <= prekks, prekks < v_bs[1:].reshape(-1, 1))
        kks = prekks.argmax(0)
        kks_values = prekks.sum(0) > 0

        values = np.uint64(np.logical_and(np.logical_and(iis_values, jjs_values), kks_values))
        np.add.at(p2v_dist, (iis, jjs, kks), values)

    return np.float64(p2v_dist) / p2v_dist.sum()



def JSD(clouds1, clouds2, clouds1_flag='gen', clouds2_flag='ref', warning=True):
    dist1 = get_voxel_occ_dist(clouds1, clouds_flag=clouds1_flag, warning=warning)
    dist2 = get_voxel_occ_dist(clouds2, clouds_flag=clouds2_flag, warning=warning)
    return entropy((dist1 + dist2).flatten() / 2.0, base=2) - \
        0.5 * (entropy(dist1.flatten(), base=2) + entropy(dist2.flatten(), base=2))



def COV(dists, axis=1):
    return float(dists.min(axis)[1].unique().shape[0]) / float(dists.shape[axis])


def MMD(dists, axis=1):
    return float(dists.min((axis + 1) % 2)[0].mean().float())


def KNN(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((-torch.ones(n0), torch.ones(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, 0).float()
    pred[torch.eq(pred, 0)] = -1.

    return float(torch.eq(label, pred).float().mean())

#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds, as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))

ZERO = 0.1
ONE = 0.9


def smooth_labels(B,ran=[0.9,1.0]):
    #return y - 0.3 + (np.random.random(y.shape) * 0.5)
    return (ran[1]-ran[0])*np.random.random(B) + ran[0]

#y = ones((n_samples, 1))
# example of smoothing class=1 to [0.7, 1.2
def smooth_positive_labels(B,ran=[0.9,1.0]):
    #return y - 0.3 + (np.random.random(y.shape) * 0.5)
    return (ran[1]-ran[0])*np.random.random((B,)) + ran[0]

# example of smoothing class=0 to [0.0, 0.3]
#y = zeros((n_samples, 1))
def smooth_negative_labels(B,ran=[0.0,0.1]):
    #return y + np.random.random(y.shape) * 0.3
    return (ran[1]-ran[0])*np.random.random((B,)) + ran[0]


# randomly flip some labels
#y = ones((n_samples, 1))
#or y = zeros((n_samples, 1))
def noisy_labels(y, p_flip=0.05):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y

def gen_loss(d_real, d_fake, gan="wgan", weight=1., d_real_p=None, d_fake_p=None,noise_label=False):
    if gan.lower() == "wgan":
        wg_loss_orig = - d_fake.mean()
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_gen_loss": wg_loss.clone().detach().item(),
            "wgan_gen_loss_orig": wg_loss_orig.clone().detach().item(),
        }
    elif gan.lower() == "hinge":
        g_loss = -d_fake.mean()
        d_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        d_acc = d_correct / float(d_real.size(0) + d_fake.size(0))

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            "dis_acc": d_acc.clone().detach(),
            "dis_correct": d_correct.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "ls":
        #mse = nn.MSELoss()
        B = d_fake.size(0)
        #real_label_np = np.ones((B,))
        fake_label_np = np.ones((B,1))

        if noise_label:
            # occasionally flip the labels when training the generator to fool the D
            fake_label_np = noisy_labels(fake_label_np, 0.05)

        #real_label = torch.from_numpy(real_label_np.astype(np.float32)).cuda()

        fake_label = torch.from_numpy(fake_label_np.astype(np.float32)).cuda()

        # real_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(1).cuda())
        # fake_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(0).cuda())
        g_loss = F.mse_loss(d_fake, fake_label)



        if d_fake_p is not None:
            fake_label_p = Variable(torch.FloatTensor(d_fake_p.size(0), d_fake_p.size(1)).fill_(1).cuda())
            g_loss_p = F.mse_loss(d_fake_p,fake_label_p)
            g_loss = g_loss + 0.2*g_loss_p

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "gan":
        fake_target = torch.tensor([1.0]).cuda()
        fake_loss = functools.partial(BCEfakeloss, target=fake_target)
        g_loss = fake_loss(d_fake)

        if d_fake_p is not None:
            g_loss_p = fake_loss(d_fake_p.view(-1))
            g_loss = g_loss + g_loss_p

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "real":
        # https://github.com/weishenho/SAGAN-with-relativistic/blob/master/main.py
        y = Variable(torch.Tensor(d_real.size(0)).fill_(1.0), requires_grad=False)
        d_loss =  torch.mean((d_real - torch.mean(d_fake) + y) ** 2)
        g_loss = torch.mean((d_fake - torch.mean(d_real) - y) ** 2)

        # d_loss = torch.mean((d_real - torch.mean(d_fake) - y) ** 2)
        # g_loss = torch.mean((d_fake - torch.mean(d_real) + y) ** 2)
        loss = (g_loss + d_loss) / 2.0

    else:
        raise NotImplementedError("Not implement: %s" % gan)


def mix_loss(d_mix, gan="wgan", weight=1.,d_mix_p=None,target_map_p=None):


    if gan.lower() == "ls":
        fake_label = Variable(torch.FloatTensor(d_mix.size(0)).fill_(0).cuda())
        mix_loss = F.mse_loss(d_mix, fake_label)

        if d_mix_p is not None:
            mix_loss_p = F.mse_loss(d_mix_p, target_map_p)
            mix_loss = (mix_loss + mix_loss_p)/2.0

        loss =  mix_loss
        return loss, {
            'loss': loss.clone().detach(),
        }
    elif gan.lower() =="gan":
        fake_target = torch.tensor([0.0]).cuda()

        mix_loss = F.binary_cross_entropy_with_logits(d_mix, fake_target.expand_as(d_mix),
                                                     reduction="none")

        if d_mix_p is not None:

            consistency_loss = F.mse_loss(d_mix_p, target_map_p)

            mix_list = []
            for i in range(d_mix_p.size(0)):
                # MIXUP LOSS 2D
                mix2d_i = F.binary_cross_entropy_with_logits(d_mix_p[i].view(-1), target_map_p[i].view(-1))
                mix_list.append(mix2d_i)

            D_loss_mixed_2d = torch.stack(mix_list)

            mix_loss = D_loss_mixed_2d + mix_loss
            mix_loss = mix_loss.mean()


            mix_loss = mix_loss + consistency_loss
            # -> D_loss_mixed_2d.mean() is taken later
        else:
            mix_loss = mix_loss.mean()

        loss = mix_loss
        return loss, {
            'loss': loss.clone().detach(),
        }
    else:
        raise NotImplementedError("Not implement: %s" % gan)

def dis_loss(d_real, d_fake, gan="wgan", weight=1.,d_real_p=None, d_fake_p=None, noise_label=False):
    # B = d_fake.size(0)
    # a = 1.0
    # b = 0.9

    if gan.lower() == "wgan":
        loss_fake = d_fake.mean()
        loss_real = d_real.mean()
        wg_loss_orig = loss_fake - loss_real
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_dis_loss": wg_loss.clone().detach().item(),
            "wgan_dis_loss_orig": wg_loss_orig.clone().detach().item(),
            "wgan_dis_loss_real": loss_real.clone().detach().item(),
            "wgan_dis_loss_fake": loss_fake.clone().detach().item()
        }
    elif gan.lower() == "hinge":
        d_loss_real = torch.nn.ReLU()(1.0 - d_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_fake).mean()

        # d_loss_real = -torch.min(d_real - 1, d_real * 0).mean()
        # d_loss_fake = -torch.min(-d_fake - 1, d_fake * 0).mean()
        real_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        real_acc = real_correct / float(d_real.size(0) + d_fake.size(0))

        d_loss = d_loss_real + d_loss_fake
        loss = d_loss * weight
        return loss, {
            "loss": loss.clone().detach(),
            "d_loss": d_loss.clone().detach(),
            "dis_acc": real_acc.clone().detach(),
            "dis_correct": real_correct.clone().detach(),
            "loss_real": d_loss_real.clone().detach(),
            "loss_fake": d_loss_fake.clone().detach(),
        }
    elif gan.lower() == "ls":
        mse = nn.MSELoss()
        B = d_fake.size(0)

        real_label_np = np.ones((B,1))
        fake_label_np = np.zeros((B,1))

        if noise_label:
            real_label_np = smooth_labels(B,ran=[0.9,1.0])
            #fake_label_np = smooth_labels(B,ran=[0.0,0.1])
            # occasionally flip the labels when training the D to
            # prevent D from becoming too strong
            real_label_np = noisy_labels(real_label_np, 0.05)
            #fake_label_np = noisy_labels(fake_label_np, 0.05)


        real_label = torch.from_numpy(real_label_np.astype(np.float32)).cuda()
        fake_label = torch.from_numpy(fake_label_np.astype(np.float32)).cuda()


        # real_label = Variable((1.0 - 0.9) * torch.rand(d_fake.size(0)) + 0.9).cuda()
        # fake_label = Variable((0.1 - 0.0) * torch.rand(d_fake.size(0)) + 0.0).cuda()

        t = 0.5
        real_correct = (d_real >= t).float().sum()
        real_acc = real_correct / float(d_real.size(0))

        fake_correct  = (d_fake < t).float().sum()
        fake_acc = fake_correct / float(d_fake.size(0))
        # + d_fake.size(0))

        # real_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(1).cuda())
        # fake_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(0).cuda())

        g_loss = F.mse_loss(d_fake, fake_label)
        d_loss = F.mse_loss(d_real, real_label)

        if d_real_p is not None and d_fake_p is not None:

            real_label_p = Variable((1.0 - 0.9) * torch.rand(d_fake_p.size(0), d_fake_p.size(1)) + 0.9).cuda()
            fake_label_p = Variable((0.1 - 0.0) * torch.rand(d_fake_p.size(0), d_fake_p.size(1)) + 0.0).cuda()

            # real_label_p = Variable(torch.FloatTensor(d_real_p.size(0), d_real_p.size(1)).fill_(1).cuda())
            # fake_label_p = Variable(torch.FloatTensor(d_real_p.size(0), d_real_p.size(1)).fill_(0).cuda())
            g_loss_p = F.mse_loss(d_fake_p, fake_label_p)
            d_loss_p = F.mse_loss(d_real_p, real_label_p)

            g_loss = (g_loss + 0.1*g_loss_p)
            d_loss = (d_loss + 0.1*d_loss_p)

        loss =  (g_loss+d_loss)/2.0
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach(),
            'd_loss': g_loss.clone().detach(),
            "fake_acc": fake_acc.clone().detach(),
            "real_acc": real_acc.clone().detach()
        }
    elif gan.lower() =="gan":
        d_real_target = torch.tensor([1.0]).cuda()
        d_fake_target = torch.tensor([0.0]).cuda()
        discriminator_loss = functools.partial(BCEloss, d_real_target=d_real_target, d_fake_target=d_fake_target)

        g_loss, d_loss = discriminator_loss(d_fake, d_real)

        if d_real_p is not None and d_fake_p is not None:
            g_loss_p,d_loss_p = discriminator_loss(d_fake_p.view(-1),d_real_p.view(-1))
            g_loss = (g_loss + g_loss_p)/2.0
            d_loss = (d_loss + d_loss_p)/2.0

        loss =  (g_loss+d_loss)/2.0
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach(),
            'd_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "real":
        y = Variable(torch.Tensor(d_real.size(0)).fill_(1.0), requires_grad=False)
        d_loss = torch.mean((d_real - torch.mean(d_fake) - y) ** 2)
        g_loss = torch.mean((d_fake - torch.mean(d_real) + y) ** 2)
        loss =  (g_loss+d_loss)/2.0

    else:
        raise NotImplementedError("Not implement: %s" % gan)

def BCEloss(D_fake, D_real, d_real_target, d_fake_target):
    real =  F.binary_cross_entropy_with_logits(D_real,d_real_target.expand_as(D_real))
    fake =  F.binary_cross_entropy_with_logits(D_fake,d_fake_target.expand_as(D_fake))
    return real, fake

def BCEfakeloss(D_fake,target):
    return F.binary_cross_entropy_with_logits(D_fake, target.expand_as(D_fake))


def dis_acc(d_real, d_fake, loss_type="wgan", **kwargs):
    if loss_type.lower() == "wgan":
        # No threshold, don't know which one is correct which is not
        return {}
    elif loss_type.lower() == "hinge":
        return {}
    else:
        raise NotImplementedError("Not implement: %s" % loss_type)


def gradient_penalty(x_real, x_fake, d_real, d_fake,
                     lambdaGP=10., gp_type='zero_center', eps=1e-8):
    if gp_type == "zero_center":
        bs = d_real.size(0)
        grad = torch.autograd.grad(
            outputs=d_real, inputs=x_real,
            grad_outputs=torch.ones_like(d_real).to(d_real),
            create_graph=True, retain_graph=True)[0]
        # [grad] should be either (B, D) or (B, #points, D)
        grad = grad.reshape(bs, -1)
        grad_norm = gp_orig = torch.sqrt(torch.sum(grad ** 2, dim=1)).mean()
        gp = gp_orig ** 2. * lambdaGP

        # real_image.requires_grad = True
        # grad_real = torch.autograd.grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
        # grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        # grad_penalty_real = 10 / 2 * grad_penalty_real
        # grad_penalty_real.backward()

        return gp, {
            'gp': gp.clone().detach().cpu(),
            'gp_orig': gp_orig.clone().detach().cpu(),
            'grad_norm': grad_norm.clone().detach().cpu()
        }
    else:
        raise NotImplemented("Invalid gp type:%s" % gp_type)


    #dist, ass = EMD(sample, ref, 0.005, 300)



if __name__ == "__main__":
    B, N = 2, 10
    x = torch.rand(B, N, 3)
    y = torch.rand(B, N, 3)

    distChamfer = CD_loss()
    min_l, min_r = distChamfer(x.cuda(), y.cuda())
    print(min_l.shape)
    print(min_r.shape)

    l_dist = min_l.mean().cpu().detach().item()
    r_dist = min_r.mean().cpu().detach().item()
    print(l_dist, r_dist)
