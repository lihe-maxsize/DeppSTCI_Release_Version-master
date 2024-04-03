from torch import set_grad_enabled
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torch


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def run_epoch(self, dataloader, device, train_mode):
        loss_counter = Counter()
        err_counter = Counter()

        progress = tqdm(total=len(dataloader), desc='LR: | Loss: | Err: ')
        self.model.train(mode=train_mode)
        with set_grad_enabled(train_mode):
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                loss, err = self.step(x, y, train_mode)
                loss_counter.append(loss)
                err_counter.append(err)
                progress.set_description(
                    f'LR: {self.scheduler.get_last_lr()[-1]:.8f} | '
                    f'Loss: {loss_counter.latest_average():.8f} | '
                    f'Err: {err_counter.latest_average():.8f}'
                )
                progress.update(1)
        return loss_counter.latest_average(), err_counter.latest_average()

    def step(self, x, y, train_mode):
        if train_mode:
            self.optimizer.zero_grad()
            self.model.zero_grad()
        num = y.shape[0]
        y = y.view(num, 1)
        y_pred = self.model(x)
        y_num = torch.round(y_pred)
        loss = self.criterion(y_pred, y)
        if train_mode:
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        err = torch.ne(y, y_num).sum()
        err = err/num
        return loss, err


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