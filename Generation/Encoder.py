import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

cudnn.benchnark=True


class BasicConv1D(nn.Module):
    def __init__(self, Fin, Fout, act=True, norm="BN", kernal=1):
        super(BasicConv1D, self).__init__()

        self.conv = nn.Conv1d(Fin,Fout,kernal)
        if act:
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = None

        if norm is not None:
            self.norm = nn.BatchNorm1d(Fout) if norm=="BN" else nn.InstanceNorm1d(Fout)
        else:
            self.norm = None

    def forward(self, x):
        x = self.conv(x)  # Bx2CxNxk

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)

        return x


class Encoder(nn.Module):
    def __init__(self, latent_size=128):
        super(Encoder, self).__init__()
        BN = True
        # self.small_d = opts.small_d
        self.latent_size = latent_size

        self.mlps = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv1d(64, 128, 1),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv1d(128, 256, 1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv1d(256, 256, 1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(1e-2, inplace=True),
        )

        self.mode = ["max","max_avg"][0]

        if self.mode == "max":
            dim = 1024
        else:
            dim = 512
        self.fc2 = nn.Sequential(
            nn.Conv1d(256,dim,1),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(1e-2, inplace=True)
        )


        self.mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LeakyReLU(1e-2, inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(1e-2, inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(1e-2, inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(64, self.latent_size)
            )

    def forward(self, x):
        B = x.size()[0]


        x = self.mlps(x)
        x = self.fc2(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)

        if self.mode == "max":
            x = x1
        else:
            x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
            x = torch.cat((x1, x2), 1)
        #x2 = x2.view(batchsize,1024)
        x3 = self.mlp(x)

        return x3
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    def resolve(self):
        for p in self.parameters():
            p.requires_grad = True

class Remapper(nn.Module):
    def __init__(self, latent_size=128):
        super(Remapper, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * latent_size, 2 * latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * latent_size, 2 * latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * latent_size, 2 * latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * latent_size, 2 * latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * latent_size, 2 * latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * latent_size, latent_size),
        )
    def forward(self, x):
        return self.mlp(x)