import yaml
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import time
import torch
from EJCNN import EJCNN
from EJCNN_Trainer import Trainer
from EJ_Dataset import EJ_Dataset

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open("config.yml"))

    model = EJCNN()
    model = model.to(device)

    optimizer = AdamW(model.parameters(), **config['optimizer'])
    scheduler = OneCycleLR(optimizer, **config['scheduler'])
    criterion = torch.nn.MSELoss(reduction='mean')

    trainer = Trainer(model, optimizer, scheduler, criterion)
    # data_x = np.load("../Data_Generator/raw_data/EJ_P_transN256_T64_O_list_data_x.npz.npy").astype(np.float32)
    data_x = np.load("../Reality_data/raw_data/EJ_P_transN20_T21_O_list_data_x.npz.npy")
    data_x = torch.from_numpy(data_x)
    # data_y = np.load("../Data_Generator/raw_data/EJ_P_transN256_T64_O_list_data_label.npz.npy").astype(np.float32)
    data_y = np.load("../Reality_data/raw_data/EJ_P_transN20_T21_O_list_data_label.npz.npy")
    data_y = torch.from_numpy(data_y)
    train_dataset, val_dataset, test_dataset = random_split(EJ_Dataset(data_x, data_y, config["data_generator"]["n"],
                                                                      config["data_generator"]["m"],
                                                                      config["data_generator"]["window"]),
                                                            [0.6, 0.1, 0.3])
    train_loader = DataLoader(train_dataset, batch_size=config["data_loader"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["data_loader"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["data_loader"]["batch_size"], shuffle=False)

    for epoch in range(config["train"]["epoch"]):
        print(f"\n***************************** epoch {epoch} ********************************\n")
        loss, err = trainer.run_epoch(train_loader, device,
                                      train_mode=True
                                      )
        if (epoch + 1) % 10 == 0:
            _, val_err = trainer.run_epoch(val_loader, device,
                                      train_mode=False)
            print(f"\n\n\n***************************** epoch {epoch}  val_err: {val_err} ********************************\n\n")

    _, test_err = trainer.run_epoch(test_loader, device,
                      train_mode=False)
    test_err = test_err.item()
    print(f"\n\n\n***************************** test_err: {test_err} ********************************\n\n")
    timestamp = int(time.time())
    torch.save(model.state_dict(), f"./training/prediction_model_reality_data_{timestamp}_err{test_err}.pth")

if __name__ == '__main__':
    main()

