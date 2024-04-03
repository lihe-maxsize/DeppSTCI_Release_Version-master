from data_generator import Data_generator
import multiprocessing
from global_variables import global_variable
import logging
import time
import pickle
import numpy as np
from data_utils import position_encode

def data_generate_API(T, M, N, C, base):
    generator = Data_generator(T=T, N=N, C=C, M=M, base=base)
    print(f"Producing_raw_data_o with para: T{T}, M{M}, N{N}, C{C}")
    generator.generating_observation()
    print(f"Success!!!! Producing_raw_data_o with para: T{T}, M{M}, N{N}, C{C}")
    print("\n********************************************\n")
    print(f"Producing_raw_data_i_with_lambda with para: T{T}, M{M}, N{N}, C{C}")
    generator.generating_intervention_with_lambda()
    print(f"Success!!!! Producing_raw_data_i_with_lambda with para: T{T}, M{M}, N{N}, C{C}")


def gv_exp(gv):
    generator = Data_generator(T=21, N=1, C=1, M=1, base="./raw_data/", gv=gv)

    with open("conflict_year_index_01.pkl", "rb") as cfif:
        conflict_data = pickle.load(cfif)
    with open("year_loss_index_01.pkl", "rb") as ylif:
        forest_loss_data = pickle.load(ylif)
    conflict_data = np.reshape(conflict_data, (22, 100))
    conflict_data_num_list = np.sum(conflict_data, axis=1)
    forest_loss_data = np.reshape(forest_loss_data, (22, 100))
    forest_loss_data_list = np.sum(forest_loss_data, axis=1)
    # try:
    data = generator.generating_observation()
    loss = 0
    for year in range(20):
        simu_num_w = len(data[0][year][2][0])
        simu_num_y = len(data[0][year][3][0])
        truth_num_w = conflict_data_num_list[year+2]  # 2003 start
        truth_num_y = forest_loss_data_list[year+2]  # 2003 start
        loss += np.abs(simu_num_w - truth_num_w) + np.abs(simu_num_y - truth_num_y)
    return loss


def data_api(T=21, N=20, C=1, M=1):
    rou0_x3 = 0.83583
    rou0_x4 = 0.738
    rou1_x4 = 0.33777
    rou1_x3 = 1.169
    alpha0 = 0.92463
    alphax = 0.125359
    alphaw = 1.39026
    alphay = 0.5757
    gama0 = 0.29174
    gamax = 0.14475
    gama2 = 0.246
    gamaw = 1.266
    gamay = 1.23657
    gv = global_variable(rou0_x3=rou0_x3, rou1_x3=rou1_x3,
                         rou0_x4=rou0_x4, rou1_x4=rou1_x4,
                         alpha0=alpha0, alphax=[alphax, alphax, alphax, alphax],
                         alphaw=alphaw, alphay=alphay,
                         gama0=gama0, gamax=[gamax, gamax, gamax, gamax],
                         gama2=gama2, gamaw=gamaw,
                         gamay=gamay)
    generator = Data_generator(T=T, N=N, C=C, M=M, base="./raw_data/", gv=gv)
    generator.generating_observation()
    generator.generating_intervention_wt_lambda()
    generator.generating_intervention_with_lambda()


def main1():
    cfM = [1, 3, 5, 7]
    cfN = [20]
    cfC = [3, 4, 5, 6, 7]
    with multiprocessing.Pool() as pool:
        pool.starmap(data_api, [(21, N, C, M) for M in cfM for N in cfN for C in cfC])
    rou0_x3 = 0.83583
    rou0_x4 = 0.738
    rou1_x4 = 0.33777
    rou1_x3 = 1.169
    alpha0 = 0.92463
    alphax = 0.125359
    alphaw = 1.39026
    alphay = 0.5757
    gama0 = 0.29174
    gamax = 0.14475
    gama2 = 0.246
    gamaw = 1.266
    gamay = 1.23657
    gv = global_variable(rou0_x3=rou0_x3, rou1_x3=rou1_x3,
                         rou0_x4=rou0_x4, rou1_x4=rou1_x4,
                         alpha0=alpha0, alphax=[alphax, alphax, alphax, alphax],
                         alphaw=alphaw, alphay=alphay,
                         gama0=gama0, gamax=[gamax, gamax, gamax, gamax],
                         gama2=gama2, gamaw=gamaw,
                         gamay=gamay)
    Train_data_gene = Data_generator(T=21, N=20, C=3, M=3, base="./raw_data/", gv=gv)
    Train_data_gene.dspp_data()
    Train_data_gene.ej_data()


def param_exp():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=f'./logging/experiment{int(time.time())}.log',
                        filemode='w')
    min_loss = 111111111111

    rou0_x3 = 0.83583
    rou0_x4 = 0.738
    rou1_x4 = 0.33777
    rou1_x3 = 1.169
    alpha0 = 0.92463
    alphax = 0.125359
    alphaw = 1.39026
    alphay = 0.5757
    gama0 = 0.29174
    gamax = 0.14475
    gama2 = 0.246
    gamaw = 1.266
    gamay = 1.23657
    try:
        gv = global_variable(rou0_x3=rou0_x3, rou1_x3=rou1_x3,
                             rou0_x4=rou0_x4, rou1_x4=rou1_x4,
                             alpha0=alpha0, alphax=[alphax, alphax, alphax, alphax],
                             alphaw=alphaw, alphay=alphay,
                             gama0=gama0, gamax=[gamax, gamax, gamax, gamax],
                             gama2=gama2, gamaw=gamaw,
                             gamay=gamay)
        loss = gv_exp(gv)
    except:
        loss = -1
    if loss != -1 and loss < min_loss:
        min_loss = loss
    print(loss)
    logging.info(f'gv:rou0_x3={rou0_x3}, rou0_x4={rou0_x4}, rou1_x4={rou1_x4}, rou1_x3={rou1_x3},  alpha0={alpha0}, alphax=[{alphax}], alphaw={alphaw}, alphay={alphay}, gama0={gama0}, gamax=[{gamax}], gama2={gama2}, gamaw={gamaw}, gamay={gamay} \nmin_loss:{min_loss} loss:{loss}\n')


if __name__ == '__main__':
    main1()
    with open("./raw_data/N20_T21_O_list.pkl", "rb") as cf:
        data = pickle.load(cf)
    seed_data = data[0]
    for year_data in seed_data:
        year_wt = year_data[2]
        year_wt_index, _ = position_encode(year_wt)
        print(year_wt_index)