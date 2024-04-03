import torch
import yaml
import numpy as np
import time
import logging
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split, TensorDataset
from DSPP import DSPP
from DSPP_Trainer import Trainer, loss_function

class model_config:
    def __init__(self, n_points=100, y_points=10, emb_dim=32, drop_out=0.1, hid_dim=128, n_head=2, n_layers=3, z_dim=128, decoder_n_layer=3):
        self.n_points = n_points
        self.y_points = y_points
        self.emb_dim = 32
        self.drop_out = 0.1
        self.hid_dim = 128
        self.n_head = 2
        self.n_layers = 3
        self.z_dim = 128
        self.decoder_n_layer = 3

class exp_config:
    def __init__(self, lr=0.0001, epoch=100, batch_size=16, max_lr=0.00025, total_steps=150000, pct_start=0.1):
        self.lr = lr
        self.scheduler = {
        "max_lr": max_lr,
        "total_steps": total_steps,
        "anneal_strategy": 'cos',
        "pct_start": pct_start}
        self.epoch = epoch
        self.batch_size = batch_size


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=f'./training_log/dspp_{int(time.time())}.log',
                        filemode='w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    beta = 0.8
    max_lr = 0.001
    total_step = 15000
    pct_start = 0.1
    batch_size = 16
    lr = 0.0001
    n_head = 16
    n_layers = 8
    decoder_n_layer = 8
    epoch = 300
    model_cf = model_config(n_head=n_head, n_layers=n_layers,
                            decoder_n_layer=decoder_n_layer)
    exp_cf = exp_config(lr=lr, batch_size=batch_size, max_lr=max_lr,
                        total_steps=total_step, pct_start=pct_start, epoch=epoch)

    model = DSPP(model_cf).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_cf.lr)
    scheduler = OneCycleLR(optimizer, **exp_cf.scheduler)
    criterion = loss_function
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                          beta=beta)

    # data = np.load("../Data_Generator/raw_data/DSPP_N256_T64_O_list_data_outcome.npz.npy")
    data = np.load("../Reality_data/raw_data/DSPP_N20_T21_O_list_data_outcome.npz.npy")
    data = torch.tensor(data.astype(np.float32))
    mask = data
    data = torch.unsqueeze(data, dim=2)
    torch_dataset = TensorDataset(data, mask)
    train_dataset, val_dataset, test_dataset = random_split(torch_dataset,
                                                            [0.8, 0.1, 0.1])
    batch_size = exp_cf.batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    best_err = 1e5
    best_model = None
    for epoch in range(exp_cf.epoch):
        print(f"\n***************************** epoch {epoch} ********************************\n")
        loss, err = trainer.run_epoch(train_loader, device, train_mode=True)
        if (epoch + 1) % 2 == 0:
            loss, err = trainer.run_epoch(val_loader, device, train_mode=False)
            print(f"\nepoch {epoch}  val_ERR: {err} \n")
            if err < best_err:
                best_err = err
                best_model = model.state_dict()
            logging.info(f"\nepoch {epoch}  best_err {best_err} val_ERR: {err}\n")
    loss, err = trainer.run_epoch(test_loader, device, train_mode=False)
    logging.info(f"\n test_ERR: {err}\n")
    timestamp = int(time.time())
    # file_name = f"./training/DSPP_model_{timestamp}.pth"
    file_name = f"./training/DSPP_model_reality_data_{err}_{timestamp}.pth"
    logging.info(f"model_path:{file_name}")
    torch.save(best_model, file_name)

if __name__ == '__main__':
    main()
