from torch import set_grad_enabled
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torch.nn as nn
import torch


def ll_func(mask, density, density2):
    ll_x = torch.sum(torch.log(density) * mask)
    ll_u = torch.sum(density2) * 0.01 - 1
    return ll_x - ll_u


def loss_function(mask, density, density2, pm, pv, beta):
    kl = torch.mean(kl_normal(pm, pv))
    eval = - ll_func(mask, density, density2)
    return beta * eval + (1 - beta) * kl


def kl_normal(pm, pv):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    qm = nn.Parameter(torch.zeros(1), requires_grad=False).to(device)
    qv = nn.Parameter(torch.ones(1), requires_grad=False).to(device)
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


def error_func(density, mask, mean_error=False):
    mask = torch.where(mask == 0, -1, mask)
    err = - density * mask
    if mean_error:
        err = torch.mean(err)
    else:
        err = torch.sum(err)
    return err

# LL loss
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, beta):
        self.beta = beta
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def run_epoch(self, dataloader, device, train_mode):
        loss_counter = Counter()
        err_counter = Counter()
        progress = tqdm(total=len(dataloader), desc='LR: | Loss: | LL: |Err:')
        self.model.train(mode=train_mode)
        with set_grad_enabled(train_mode):
            for batch in dataloader:
                x, mask = batch
                x = x.to(device)
                mask = mask.to(device)
                loss, err, ll = self.step(x, mask, train_mode)
                loss_counter.append(loss)
                err_counter.append(err)
                progress.set_description(
                    f'LR: {self.scheduler.get_last_lr()[-1]:.8f} | '
                    f'Loss: {loss_counter.latest_average():.8f} | '
                    f'LL: {ll:.8f} |'
                    f'Err: {err:.8f}'
                )
                progress.update(1)
        return loss_counter.latest_average(), err_counter.latest_average()

    def step(self, x, mask, train_mode):
        if train_mode:
            self.optimizer.zero_grad()
            self.model.zero_grad()
        pm, pv, density = self.model(x)
        u = torch.ones_like(x)
        _, _, density2 = self.model(u)
        loss = self.criterion(mask, density, density2, pm, pv, self.beta)
        density_num = torch.sum(density, dim=1) * 0.01  # 这里应该是要乘以0.01的，因为density的大小是0-100
        mask_num = torch.sum(mask, dim=1)
        err = torch.nn.MSELoss(reduction="mean")(density_num, mask_num)
        loss = loss + 10 * err
        ll = ll_func(mask, density, density2)
        if train_mode:
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
        return loss, err, ll


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
