import torch
import pickle
import numpy as np
import math
from tqdm import tqdm
import logging
import time
from Data_Generator.data_utils import position_encode
from Data_Generator.confounder_generator import mesh_X1_X2
from EJCNN import EJCNN
from DSPP import DSPP
from DSPP_main import model_config

def EJ_func(model, WJ, Ht, intervention_C):
    x_lamb = int(torch.round(model(Ht)).item())
    x_lamb = intervention_C * math.log(x_lamb)
    this_k = len(WJ[0])
    EJ = (x_lamb ** this_k / math.factorial(this_k)) * math.exp(- x_lamb)
    return EJ

def fh_WJ_func(this_lamb, this_WJ):
    """
    :param this_lamb:  float 64
    :param this_WJ:  list2 : w_x, w_y
    :return: fh
    """
    this_k = len(this_WJ[0])
    fh = (this_lamb ** this_k / math.factorial(this_k)) * math.exp(- this_lamb)
    return fh


def Truth_NB_func(N, T, M, C):
    file_path = f"./Data_Generator/raw_data/N{N}_T{T}_M{M}_C{C}_I_lambda.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    NB_list = []
    progress = tqdm(total=N, desc=f'Truth_NB: N:{N}, T:{T}, M{M}, C{C}, S: ', position=0)
    for S in range(N):
        NBt_list = []
        for It in range(2, T-M):
            i_key = f"N{N}_T{T}_S{S}_M{M}_It{It}_C{C}_I"
            Y_data = data[i_key][It + M - 2][3][0]
            NBt = len(Y_data)
            NBt_list.append(NBt)
        NB = np.mean(np.array(NBt_list))
        NB_list.append(NB)
        progress.set_description(
            f'Truth_NB: N:{N}, T:{T}, M{M}, C{C}, S:{S} '
        )
        progress.update(1)
    NB_res = np.mean(np.array(NB_list))
    return NB_res

def Yt_est_func(i_l_data, o_data, dspp_model, pmodel, device, this_t, N, T, M, C, seed, window=5):
    """
    :param this_t: the time stamp for Yt
    :param N:
    :param T: total time
    :param M: intervention length
    :param C: intervention param
    :param seed:
    :return:
    """

    fh_list = []
    EJ_list = []
    it = this_t-M + 1
    for j in range(this_t-M+1, this_t+1):
        WJ_key = f"N{N}_T{T}_S{seed}_M{M}_It{it}_C{C}_I"
        lambda_key = WJ_key + "_lambda"
        this_lamb = i_l_data[lambda_key][str(j)]
        WJ = i_l_data[WJ_key][j - 1][2]
        fh = fh_WJ_func(this_lamb, WJ)
        fh = max(fh, 0.01)
        fh_list.append(fh)
        o_key = f"N{N}_T{T}_S{seed}_O"
        i_key = f"N{N}_T{T}_S{seed}_M{M}_It{it}_C{C}_I"
        i_stamp = i_l_data[i_key]
        WJ = i_stamp[j - 1][2]
        Ht = i_stamp[j - window - 1:j]
        Ht_index = []
        for index, stamp in enumerate(Ht):
            x3 = list(stamp[0].reshape(100))
            x4 = list(stamp[1].reshape(100))
            W, _ = position_encode(stamp[2], n=10, m=10)
            Y, _ = position_encode(stamp[3], n=10, m=10)
            if index == 0:
                Ht_index.extend([W, Y])
            elif index == window:
                Ht_index.extend([x3, x4])
            else:
                Ht_index.extend([x3, x4, W, Y])
        X1, X2 = mesh_X1_X2(10, 10)
        Ht_index.append(list(X1))
        Ht_index.append(list(X2))
        Ht_index = torch.tensor(Ht_index)
        Ht_index = torch.unsqueeze(Ht_index, dim=0)
        Ht_index = Ht_index.to(device)
        Ht_index = Ht_index.type(torch.float32)
        Ht_index = Ht_index.view(1, 22, 10, 10)
        EJ = EJ_func(pmodel, WJ, Ht_index, C)
        EJ = max(EJ, 0.01)
        EJ_list.append(EJ)
    first_item = torch.prod(torch.tensor(fh_list) / torch.tensor(EJ_list))

    SYt_key = f"N{N}_T{T}_S{seed}_O"
    SYt = o_data[SYt_key][this_t-1][3]
    SYt_index, _ = position_encode(SYt, n=10, m=10)
    SYt_index = torch.tensor(SYt_index, dtype=torch.float32)
    SYt_index = SYt_index.view(1, 100, 1).to(device)

    _, _, density = dspp_model(SYt_index)
    density_num = torch.sum(density) * 0.01
    Yt_est = first_item * density_num
    NBt_est = torch.sum(Yt_est)
    return Yt_est, NBt_est


def NB_est_func(i_l_data, o_data, dspp_model, pmodel, device, N, T, M, C, seed, window=5):
    """

    :param o_data:
    :param i_l_data:
    :param dspp_model:
    :param pmodel:
    :param device:
    :param N:
    :param T:
    :param M:
    :param C:
    :param seed:
    :param window:
    :return:
    """
    NBt_list = []
    for this_t in range(window+M, T+1):
        Yt_est, NBt_est = Yt_est_func(i_l_data, o_data, dspp_model, pmodel, device, this_t, N, T, M, C, seed)
        if NBt_est <= 100:
            NBt_list.append(NBt_est)
    NB_est = torch.mean(torch.tensor(NBt_list)).item()
    if NB_est == torch.nan:
        NB_est = -1
    return NB_est

def CI_NB_est(dspp_model, pmodel, device, N, T, M, C):
    i_l_file = f"./Data_Generator/raw_data/N{N}_T{T}_M{M}_C{C}_I_lambda.pkl"
    with open(i_l_file, "rb") as i_f:
        i_l_data = pickle.load(i_f)
    o_file = f"./Data_Generator/raw_data/N{N}_T{T}_O_dict.pkl"
    with open(o_file, "rb") as o_f:
        o_data = pickle.load(o_f)
    NB_list = []
    progress = tqdm(total=N, desc=f'CI_NB_est: N:{N}, T:{T}, M{M}, C{C}, S: ', position=0)
    for seed in range(N):
        NB_s = NB_est_func(i_l_data, o_data, dspp_model, pmodel, device, N, T, M, C, seed)
        if NB_s != -1:
            NB_list.append(NB_s)
        progress.set_description(
            f'CI_NB_est: N:{N}, T:{T}, M{M}, C{C}, S:{seed} '
        )
        progress.update(1)
    NB = torch.mean(torch.tensor(NB_list)).item()
    return NB

def Experiment():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=f'./logging/experiment{int(time.time())}.log',
                        filemode='w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pmodel = EJCNN().to(device)
    pmodel.load_state_dict(torch.load("./EJCNN/training/prediction_model_batch64.pth"))

    n_head = 16
    n_layers = 8
    decoder_n_layer = 8
    model_cf = model_config(n_head=n_head, n_layers=n_layers,
                            decoder_n_layer=decoder_n_layer)
    dspp_model = DSPP(model_cf).to(device)
    dspp_model.load_state_dict(torch.load("./DSPP/training/DSPP_model_1693876519_err 0.32.pth"))

    cfT = [64, 48, 32]
    cfM = [1, 3, 5, 7]
    cfN = [20]
    cfC = [3, 4, 5, 6, 7]
    for N in cfN:
        for T in cfT:
            for M in cfM:
                for C in cfC:
                    Truth_NB = Truth_NB_func(N, T, M, C)
                    print("NB_truth", Truth_NB)
                    logging.info(f'Truth_NB:{Truth_NB} N:{N}, T:{T}, M{M}, C{C}')
                    CI_NB = CI_NB_est(dspp_model, pmodel, device, N, T, M, C)
                    print("NB_est", CI_NB)
                    logging.info(f'CI_NB_est:{CI_NB} N:{N}, T:{T}, M{M}, C{C}')


def Reality_NB_count():
    with open("./Reality_data/raw_data/N20_T21_O_list.pkl", "rb") as rf:
        data = pickle.load(rf)
    NB_list_by_T = []
    for seed_data in data:
        seed_Nlist = []
        for time_stamp in seed_data:
            Yt = len(time_stamp[3][0])
            seed_Nlist.append(Yt)
        NB_list_by_T.append(np.array(seed_Nlist))
    # NB_list_by_T 20 * 21
    NB_list_by_T = np.array(NB_list_by_T)
    NB_list_by_T = np.mean(NB_list_by_T, axis=0)
    NB = np.mean(np.array(NB_list_by_T))
    return NB, NB_list_by_T


def Yt_est_func_R(i_l_data, o_data, dspp_model, pmodel, device, this_t, N, T, M, C, seed, window=5):
    """
    :param this_t: the time stamp for Yt
    :param N:
    :param T: total time
    :param M: intervention length
    :param C: intervention param
    :param seed:
    :return:
    """

    fh_list = []
    EJ_list = []
    it = this_t-M + 1
    for j in range(this_t-M+1, this_t+1):
        WJ_key = f"N{N}_T{T}_S{seed}_M{M}_It{it}_C{C}_"
        lambda_key = WJ_key + "_lambda"
        this_lamb = i_l_data[lambda_key][str(j)]
        WJ_key_new = WJ_key + "wt"
        WJ = i_l_data[WJ_key_new][str(j)]
        fh = fh_WJ_func(this_lamb, WJ)
        fh = max(fh, 0.01)
        fh_list.append(fh)
        o_key = f"N{N}_T{T}_S{seed}_O"
        # i_key = f"N{N}_T{T}_S{seed}_M{M}_It{it}_C{C}_I"
        i_stamp = o_data[o_key]
        # WJ = i_stamp[j - 1][2]
        Ht = i_stamp[j - window - 1:j]
        Ht_index = []
        for index, stamp in enumerate(Ht):
            x3 = list(stamp[0].reshape(100))
            x4 = list(stamp[1].reshape(100))
            W, _ = position_encode(stamp[2], n=10, m=10)
            Y, _ = position_encode(stamp[3], n=10, m=10)
            if index == 0:
                Ht_index.extend([W, Y])
            elif index == window:
                Ht_index.extend([x3, x4])
            else:
                Ht_index.extend([x3, x4, W, Y])
        X1, X2 = mesh_X1_X2(10, 10)
        Ht_index.append(list(X1))
        Ht_index.append(list(X2))
        Ht_index = torch.tensor(Ht_index)
        Ht_index = torch.unsqueeze(Ht_index, dim=0)
        Ht_index = Ht_index.to(device)
        Ht_index = Ht_index.type(torch.float32)
        Ht_index = Ht_index.view(1, 22, 10, 10)
        EJ = EJ_func(pmodel, WJ, Ht_index, C)
        EJ = max(EJ, 0.01)
        EJ_list.append(EJ)
    first_item = torch.prod(torch.tensor(fh_list) / torch.tensor(EJ_list))

    SYt_key = f"N{N}_T{T}_S{seed}_O"
    SYt = o_data[SYt_key][this_t-1][3]
    SYt_index, _ = position_encode(SYt, n=10, m=10)
    SYt_index = torch.tensor(SYt_index, dtype=torch.float32)
    SYt_index = SYt_index.view(1, 100, 1).to(device)

    _, _, density = dspp_model(SYt_index)
    density_num = torch.sum(density) * 0.01
    Yt_est = first_item * density_num
    NBt_est = torch.sum(Yt_est)
    return Yt_est, NBt_est



def NB_est_func_R(i_l_data, o_data, dspp_model, pmodel, device, N, T, M, C, seed, window=5):
    """

    :param o_data:
    :param i_l_data:
    :param dspp_model:
    :param pmodel:
    :param device:
    :param N:
    :param T:
    :param M:
    :param C:
    :param seed:
    :param window:
    :return:
    """
    NBt_list = []
    for this_t in range(window+M, T+1):
        Yt_est, NBt_est = Yt_est_func_R(i_l_data, o_data, dspp_model, pmodel, device, this_t, N, T, M, C, seed)
        if NBt_est <= 100:
            NBt_list.append(NBt_est)
        else:
            NBt_list.append(-1)
    NB_est = torch.mean(torch.tensor(NBt_list)).item()
    if NB_est == torch.nan:
        NB_est = -1
    return NB_est, NBt_list


def CI_NB_est_R(dspp_model, pmodel, device, N, T, M, C):
    i_l_file = f"./Reality_data/raw_data/N{N}_T{T}_M{M}_C{C}_I_wt_lambda.pkl"
    with open(i_l_file, "rb") as i_f:
        i_l_data = pickle.load(i_f)
    o_file = f"./Reality_data/raw_data/N{N}_T{T}_O_dict.pkl"
    with open(o_file, "rb") as o_f:
        o_data = pickle.load(o_f)
    NB_list = []
    NB_year_list = []
    progress = tqdm(total=N, desc=f'CI_NB_est: N:{N}, T:{T}, M{M}, C{C}, S: ', position=0)
    for seed in range(N):
        NB_s, NB_year = NB_est_func_R(i_l_data, o_data, dspp_model, pmodel, device, N, T, M, C, seed)
        if NB_s != -1:
            NB_list.append(NB_s)
        progress.set_description(
            f'CI_NB_est: N:{N}, T:{T}, M{M}, C{C}, S:{seed} '
        )
        progress.update(1)
        NB_year_list.append(NB_year)
    NB_year_by_T = torch.mean(torch.tensor(NB_year_list), dim=0)
    NB_year_by_T = NB_year_by_T.numpy().tolist()
    NB = torch.mean(torch.tensor(NB_list)).item()
    return NB, NB_year_by_T



def Experiment_R():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=f'./logging/experiment_r_{int(time.time())}.log',
                        filemode='w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pmodel = EJCNN().to(device)
    pmodel.load_state_dict(torch.load("./EJCNN/training/prediction_model_reality_data_1694654984_err0.0.pth"))

    n_head = 16
    n_layers = 8
    decoder_n_layer = 8
    model_cf = model_config(n_head=n_head, n_layers=n_layers,
                            decoder_n_layer=decoder_n_layer)
    dspp_model = DSPP(model_cf).to(device)
    dspp_model.load_state_dict(torch.load("./DSPP/training/DSPP_model_reality_data_err0.24.pth"))

    Reality_NB, Reality_NB_list_by_T = Reality_NB_count()
    logging.info(f'Reality_NB:{Reality_NB} N:{20}, T:{21} Reality_NB_year_by_T:{Reality_NB_list_by_T}')
    cfT = [21]
    cfM = [1, 3, 5, 7]
    cfN = [20]
    cfC = [3, 4, 5, 6, 7]
    for N in cfN:
        for T in cfT:
            for M in cfM:
                for C in cfC:
                    CI_NB, CI_NB_year_by_T = CI_NB_est_R(dspp_model, pmodel, device, N, T, M, C)
                    print("NB_est", CI_NB)
                    logging.info(f'CI_NB_est:{CI_NB} N:{N}, T:{T}, M{M}, C{C} CI_NB_year_by_T:{CI_NB_year_by_T}')



def CI_NB_est_R_est(dspp_model, pmodel, device, N, T, M, C):
    i_l_file = f"./Reality_data/raw_data/N{N}_T{T}_M{M}_C{C}_I_lambda.pkl"
    with open(i_l_file, "rb") as i_f:
        i_l_data = pickle.load(i_f)
    o_file = f"./Reality_data/raw_data/N{N}_T{T}_O_dict.pkl"
    with open(o_file, "rb") as o_f:
        o_data = pickle.load(o_f)
    NB_list = []
    progress = tqdm(total=N, desc=f'CI_NB_est: N:{N}, T:{T}, M{M}, C{C}, S: ', position=0)
    for seed in range(N):
        NB_s = NB_est_func(i_l_data, o_data, dspp_model, pmodel, device, N, T, M, C, seed)
        if NB_s != -1:
            NB_list.append(NB_s)
        progress.set_description(
            f'CI_NB_est: N:{N}, T:{T}, M{M}, C{C}, S:{seed} '
        )
        progress.update(1)
    NB = torch.mean(torch.tensor(NB_list)).item()
    return NB


def Truth_NB_func_R_est(N, T, M, C):
    file_path = f"./Reality_data/raw_data/N{N}_T{T}_M{M}_C{C}_I_lambda.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    NB_list = []
    progress = tqdm(total=N, desc=f'Truth_NB: N:{N}, T:{T}, M{M}, C{C}, S: ', position=0)
    for S in range(N):
        NBt_list = []
        for It in range(2, T-M):
            i_key = f"N{N}_T{T}_S{S}_M{M}_It{It}_C{C}_I"
            Y_data = data[i_key][It + M - 2][3][0]
            NBt = len(Y_data)
            NBt_list.append(NBt)
        NB = np.mean(np.array(NBt_list))
        NB_list.append(NB)
        progress.set_description(
            f'Truth_NB: N:{N}, T:{T}, M{M}, C{C}, S:{S} '
        )
        progress.update(1)
    NB_res = np.mean(np.array(NB_list))
    return NB_res

def Experiment_R_est():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=f'./logging/experiment_r_est_{int(time.time())}.log',
                        filemode='w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pmodel = EJCNN().to(device)
    pmodel.load_state_dict(torch.load("./EJCNN/training/prediction_model_reality_data_1694654984_err0.0.pth"))

    n_head = 16
    n_layers = 8
    decoder_n_layer = 8
    model_cf = model_config(n_head=n_head, n_layers=n_layers,
                            decoder_n_layer=decoder_n_layer)
    dspp_model = DSPP(model_cf).to(device)
    dspp_model.load_state_dict(torch.load("./DSPP/training/DSPP_model_reality_data_err0.24.pth"))

    cfT = [21]
    cfM = [1, 3, 5, 7]
    cfN = [20]
    cfC = [3, 4, 5, 6, 7]
    for N in cfN:
        for T in cfT:
            for M in cfM:
                for C in cfC:
                    Truth_NB = Truth_NB_func_R_est(N, T, M, C)
                    print("NB_truth", Truth_NB)
                    logging.info(f'Truth_NB:{Truth_NB} N:{N}, T:{T}, M{M}, C{C}')
                    CI_NB = CI_NB_est_R_est(dspp_model, pmodel, device, N, T, M, C)
                    print("NB_est", CI_NB)
                    logging.info(f'CI_NB_est:{CI_NB} N:{N}, T:{T}, M{M}, C{C}')


if __name__ == '__main__':
    Experiment()
    Experiment_R()
