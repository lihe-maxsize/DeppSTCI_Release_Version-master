import torch
import torch.nn as nn
from torch.nn import functional as F
import math



class DSPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.pos_embedding
        self.encoder = Encoder(config)
        self.density_decoder = Decoder(config, softmax=False)


    def init_weight(self):
        pass

    def forward(self, x):
        # density 范围 0 - 100
        # 输入 x 维度 batch, 100, 1
        pm, pv = self.encoder(x)  # m,v -> (n_points, z_dim)
        latent_z = gauss_sample(pm, pv)  # latent_z -> (n_points, z_dim)
        density = self.density_decoder(latent_z)  # w -> (n_points, n_points)
        return pm, pv, density * 100


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置batch_first = True, 数据格式应该是(batch, seq, feature)
        self.feature = 2
        self.emb_dim = config.emb_dim
        self.embedding = nn.Linear(1, config.emb_dim, bias=False)
        self.transformer_layer = nn.TransformerEncoderLayer(nhead=config.n_head,
                                                            dim_feedforward=config.hid_dim,
                                                            d_model=config.emb_dim,
                                                            dropout=config.drop_out,
                                                            batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, config.n_layers)
        self.feedforward = nn.Linear(config.emb_dim, config.z_dim * 2)
        self.init_weight()

    def init_weight(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)  # uniform_
        self.feedforward.weight.data.uniform_(-0.1, 0.1)
        self.feedforward.bias.data.zero_()

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.emb_dim)
        output1 = self.transformer(x)
        output_z = self.feedforward(output1)
        z_prop = output_z[:, -1, :]
        m, v_ = torch.split(z_prop, z_prop.size(-1) // 2, dim=-1)
        v = F.softplus(v_) + 1e-5
        return m, v


class Decoder(nn.Module):
    def __init__(self, config, softmax=False):
        super().__init__()
        self.z_dim = config.z_dim
        self.softmax = softmax
        self.ffd = nn.Sequential(
            nn.Linear(config.z_dim, config.hid_dim),
            nn.ELU(),
            *[nn.Linear(config.hid_dim, config.hid_dim),
              nn.ELU()] * (config.decoder_n_layer - 1),
            nn.Linear(config.hid_dim, config.n_points),
        )

    def init_weight(self):
        pass

    def forward(self, z):
        output = self.ffd(z)
        output = F.softplus(output) + 1e-5
        return output


def gauss_sample(m, v):
    z = torch.randn_like(m)
    z = z * torch.sqrt(v) + m
    return z

class Counter:
    def __init__(self, limit=10):
        self.latest_data = []
        self.latest_data_total = 0
        self.previous_data_num = 0
        self.previous_data_total = 0
        self.limit = limit

    def append(self, x):
        self.latest_data.append(x)
        self.latest_data_total += x
        if len(self.latest_data) > self.limit:
            discard = self.latest_data.pop(0)
            self.latest_data_total -= discard
            self.previous_data_total += discard
            self.previous_data_num += 1

    def all_average(self):
        total = (self.latest_data_total + self.previous_data_total)
        count = (self.previous_data_num + len(self.latest_data))
        res = total / count if count > 0 else 0
        return res

    def latest_average(self):
        total = self.latest_data_total
        count = len(self.latest_data)
        res = total / count if count > 0 else 0
        return res