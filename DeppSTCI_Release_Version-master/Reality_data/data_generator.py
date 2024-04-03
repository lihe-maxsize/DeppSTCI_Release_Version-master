import random
import numpy as np
import pickle
import os
from tqdm import tqdm
from deprecated import deprecated
# from global_variables import *
from confounder_generator import get_relative_points, get_X1_array, get_X2_array, get_X3_array, get_X4_array, mesh_X1_X2
from lambda_func import lambda_o_array, lambda_w_array
from data_utils import position_encode


def data_generate_func(gv, intervention_time=None, seed=1, max_time=10, intervention_C=3, n=10, m=10, with_lambda=False):
    """
    generate observational and intervention data
    :param gv: global variable
    :param intervention_time: intervention time list
    :param seed: random seed
    :param max_time: total time
    :param intervention_C: param for intervention density
    :param n: x-resolution
    :param m: y-resolution
    :param with_lambda: record intervention lambda or not
    :return:
    raw_data - max_time * [X3, X4, [w_x, w_y], [o_x, o_y]]
    lambda_res - lambda_res
    """
    if intervention_time is None:
        intervention_time = []
    if 1 in intervention_time:
        raise ValueError("error intervention at time 1")
    random.seed(seed)
    np.random.seed(seed)
    time_step = 1
    SW_pre = gv.SW0
    SY_pre = gv.SY0
    SW_pre4 = [gv.SW0]
    lambda_max = 1500  # lambda_max should be greater than all lambda_w and lambda_y
    # for intervention data calculation
    wx_next, wy_next, NP_next, X_1w_next, X_2w_next, X_3w_next, X_4w_next, SW, SY, raw_data = [[] for i in range(10)]
    lambda_res = {}
    while (time_step <= max_time):
        data_stamp = []
        # relative points for x3 and x4
        X3_relative_points = get_relative_points(gv.rou0_x3, gv.rou1_x3, gv)
        X4_relative_points = get_relative_points(gv.rou0_x4, gv.rou1_x4, gv)
        # X3 and X4 for the whole map
        xx = np.linspace(0, 0.9, n)
        xx = xx + 0.05
        yy = np.linspace(0, 0.9, m)
        w_xx, w_yy = np.meshgrid(xx, yy)
        w_xx = np.reshape(w_xx, newshape=(n * m))
        w_yy = np.reshape(w_yy, newshape=(n * m))
        X_3ww = get_X3_array(X3_relative_points, w_xx, w_yy)
        X_4ww = get_X4_array(X4_relative_points, w_xx, w_yy)
        X3_data = np.reshape(X_3ww, newshape=(n, m))
        X4_data = np.reshape(X_4ww, newshape=(n, m))
        data_stamp.append(X3_data)
        data_stamp.append(X4_data)
        # intervention w value
        if time_step in intervention_time:
            NP = NP_next
            w_x = wx_next
            w_y = wy_next
            data_stamp.append([w_x, w_y])
        # observation w value
        else:
            # Homogeneous Poisson sampling
            NP = np.random.poisson(lam=lambda_max)
            w_x = np.random.uniform(low=0, high=1, size=NP)
            w_x = np.around(np.floor(w_x * 10) / 10 + 0.05, 2)
            w_y = np.random.uniform(low=0, high=1, size=NP)
            w_y = np.around(np.floor(w_y * 10) / 10 + 0.05, 2)
            # confounders
            X_1w = get_X1_array(gv, w_x, w_y)
            X_2w = get_X2_array(gv, w_x, w_y)
            X_3w = get_X3_array(X3_relative_points, w_x, w_y)
            X_4w = get_X4_array(X4_relative_points, w_x, w_y)

            # Non-homogeneous Poisson distribution
            lambda_w = lambda_w_array(gv.alpha0, gv.alphax, [X_1w, X_2w, X_3w, X_4w], gv.alphaw, SW_pre, gv.alphay,
                                      SY_pre, w_x, w_y)
            # reject sampling
            flag_ix = np.where(1 < lambda_w / lambda_max)
            if flag_ix[0].size > 0:
                # print(flag_ix, lambda_w)
                raise ValueError("lambdx_max is too small")
            ix = np.where(np.random.uniform(low=0, high=1, size=NP) <= lambda_w / lambda_max)
            if ix[0].size == 0:
                ix = [0]
            w_x = w_x[ix]
            w_y = w_y[ix]
            data_stamp.append([w_x, w_y])

        # Prepare data for the next intervention in advance
        if time_step + 1 in intervention_time:
            NP_next = np.random.poisson(lam=lambda_max)
            wx_next = np.random.uniform(low=0, high=1, size=NP_next)
            wx_next = np.around(np.floor(wx_next * 10) / 10 + 0.05, 2)
            wy_next = np.random.uniform(low=0, high=1, size=NP_next)
            wy_next = np.around(np.floor(wy_next * 10) / 10 + 0.05, 2)
            X_1w_next = get_X1_array(gv, wx_next, wy_next)
            X_2w_next = get_X2_array(gv, wx_next, wy_next)
            X_3w_next = get_X3_array(X3_relative_points, wx_next, wy_next)
            X_4w_next = get_X4_array(X4_relative_points, wx_next, wy_next)
            # intervention_lambda for the whole map
            if with_lambda:
                lambda_x = np.arange(0, 1, 0.1)
                lambda_y = np.arange(0, 1, 0.1)
                lambda_x, lambda_y = np.meshgrid(lambda_x, lambda_y)
                lambda_x = lambda_x.reshape(100)
                lambda_y = lambda_y.reshape(100)
                lambda_X_1w_next = get_X1_array(gv, lambda_x, lambda_y)
                lambda_X_2w_next = get_X2_array(gv, lambda_x, lambda_y)
                lambda_X_3w_next = get_X3_array(X3_relative_points, lambda_x, lambda_y)
                lambda_X_4w_next = get_X4_array(X4_relative_points, lambda_x, lambda_y)
                lambda_next = lambda_w_array(gv.alpha0, gv.alphax,
                                             [lambda_X_1w_next, lambda_X_2w_next, lambda_X_3w_next, lambda_X_4w_next],
                                             gv.alphaw, SW_pre, gv.alphay, SY_pre, lambda_x, lambda_y)
                lambda_next = intervention_C * np.log(lambda_next)
                lambda_next = np.maximum(lambda_next, 0)
                # Each point represents an area of 0.01, and integral is the same as mean
                this_lambda = np.mean(lambda_next)
                lambda_res[f"{time_step + 1}"] = this_lambda
            # non-homogeneous lambda for sampling points
            lambda_w_next = lambda_w_array(gv.alpha0, gv.alphax, [X_1w_next, X_2w_next, X_3w_next, X_4w_next], gv.alphaw, SW_pre,
                                           gv.alphay, SY_pre, wx_next, wy_next)
            lambda_w_next = intervention_C * np.log(lambda_w_next)
            lambda_w_next = np.maximum(lambda_w_next, 0)
            flag_ix_next = np.where(1 < lambda_w_next / lambda_max)
            if flag_ix_next[0].size > 0:
                # print(flag_ix_next, lambda_w_next)
                raise ValueError("lambdx_max is too small")
            ix_next = np.where(np.random.uniform(low=0, high=1, size=NP_next) <= lambda_w_next / lambda_max)
            if ix_next[0].size == 0:
                ix_next = [0]
            wx_next = wx_next[ix_next]
            wy_next = wy_next[ix_next]

        # reject sampling for outcome
        o_x = np.random.uniform(low=0, high=1, size=NP)
        o_x = np.around(np.floor(o_x * 10) / 10 + 0.05, 2)
        o_y = np.random.uniform(low=0, high=1, size=NP)
        o_y = np.around(np.floor(o_y * 10) / 10 + 0.05, 2)
        X_1o = get_X1_array(gv, o_x, o_y)
        X_2o = get_X2_array(gv, o_x, o_y)
        X_3o = get_X3_array(X3_relative_points, o_x, o_y)
        X_4o = get_X4_array(X4_relative_points, o_x, o_y)
        SW_pre = [w_x, w_y]
        SW_pre4.append(SW_pre)
        if len(SW_pre4) > 4:
            SW_pre4.pop(0)
        lambda_o = lambda_o_array(gv.gama0, gv.gamax, [X_1o, X_2o, X_3o, X_4o], gv.gama2, X_2o, gv.gamaw, SW_pre4, gv.gamay, SY_pre,
                                  o_x, o_y)
        flag_ix = np.where(1 < lambda_o / lambda_max)
        if flag_ix[0].size > 0:
            # print(flag_ix, lambda_o)
            raise ValueError("lambdx_max is too small")
        ix = np.where(np.random.uniform(low=0, high=1, size=NP) <= lambda_o / lambda_max)
        if ix[0].size == 0:
            ix = [0]
        o_x = o_x[ix]
        o_y = o_y[ix]
        data_stamp.append([o_x, o_y])
        SY_pre = [o_x, o_y]
        time_step += 1
        raw_data.append(data_stamp)
    return raw_data, lambda_res


def intervention_wt_lambda_func(gv, intervention_time=None, seed=1, max_time=10, intervention_C=3, n=10, m=10, with_lambda=False):
    """
    generate observational and intervention data
    :param gv: global variable
    :param intervention_time: intervention time list
    :param seed: random seed
    :param max_time: total time
    :param intervention_C: param for intervention density
    :param n: x-resolution
    :param m: y-resolution
    :param with_lambda: record intervention lambda or not
    :return:
    raw_data - max_time * [X3, X4, [w_x, w_y], [o_x, o_y]]
    lambda_res - lambda_res
    """
    if intervention_time is None:
        intervention_time = []
    if 1 in intervention_time:
        raise ValueError("error intervention at time 1")
    random.seed(seed)
    np.random.seed(seed)
    time_step = 1
    SW_pre = gv.SW0
    SY_pre = gv.SY0
    SW_pre4 = [gv.SW0]
    lambda_max = 1500  # lambda_max should be greater than all lambda_w and lambda_y
    # for intervention data calculation
    wx_next, wy_next, NP_next, X_1w_next, X_2w_next, X_3w_next, X_4w_next, SW, SY, raw_data = [[] for i in range(10)]
    lambda_res = {}
    wt_res = {}
    while (time_step <= max_time):
        data_stamp = []
        # relative points for x3 and x4
        X3_relative_points = get_relative_points(gv.rou0_x3, gv.rou1_x3, gv)
        X4_relative_points = get_relative_points(gv.rou0_x4, gv.rou1_x4, gv)
        # X3 and X4 for the whole map
        xx = np.linspace(0, 0.9, n)
        xx = xx + 0.05
        yy = np.linspace(0, 0.9, m)
        w_xx, w_yy = np.meshgrid(xx, yy)
        w_xx = np.reshape(w_xx, newshape=(n * m))
        w_yy = np.reshape(w_yy, newshape=(n * m))
        X_3ww = get_X3_array(X3_relative_points, w_xx, w_yy)
        X_4ww = get_X4_array(X4_relative_points, w_xx, w_yy)
        X3_data = np.reshape(X_3ww, newshape=(n, m))
        X4_data = np.reshape(X_4ww, newshape=(n, m))
        data_stamp.append(X3_data)
        data_stamp.append(X4_data)
        # intervention w value
        # anyway, generate observation data
        # Homogeneous Poisson sampling
        NP = np.random.poisson(lam=lambda_max)
        w_x = np.random.uniform(low=0, high=1, size=NP)  # 重点修改坐标点， 只能是0.05, 0.15, 0.25, .... 0.95
        w_x = np.around(np.floor(w_x * 10) / 10 + 0.05, 2)
        w_y = np.random.uniform(low=0, high=1, size=NP)  # 重点修改坐标点
        w_y = np.around(np.floor(w_y * 10) / 10 + 0.05, 2)
        # confounders
        X_1w = get_X1_array(gv, w_x, w_y)
        X_2w = get_X2_array(gv, w_x, w_y)
        X_3w = get_X3_array(X3_relative_points, w_x, w_y)
        X_4w = get_X4_array(X4_relative_points, w_x, w_y)

        # Non-homogeneous Poisson distribution
        lambda_w = lambda_w_array(gv.alpha0, gv.alphax, [X_1w, X_2w, X_3w, X_4w], gv.alphaw, SW_pre, gv.alphay, SY_pre, w_x,
                                  w_y)
        # reject sampling
        flag_ix = np.where(1 < lambda_w / lambda_max)
        if flag_ix[0].size > 0:
            # print(flag_ix, lambda_w)
            raise ValueError("lambdx_max is too small")
        ix = np.where(np.random.uniform(low=0, high=1, size=NP) <= lambda_w / lambda_max)
        if ix[0].size == 0:
            ix = [0]
        w_x = w_x[ix]
        w_y = w_y[ix]
        data_stamp.append([w_x, w_y])

        # reject sampling for outcome
        o_x = np.random.uniform(low=0, high=1, size=NP)
        o_x = np.around(np.floor(o_x * 10) / 10 + 0.05, 2)
        o_y = np.random.uniform(low=0, high=1, size=NP)
        o_y = np.around(np.floor(o_y * 10) / 10 + 0.05, 2)
        X_1o = get_X1_array(gv, o_x, o_y)
        X_2o = get_X2_array(gv, o_x, o_y)
        X_3o = get_X3_array(X3_relative_points, o_x, o_y)
        X_4o = get_X4_array(X4_relative_points, o_x, o_y)
        SW_pre = [w_x, w_y]
        SW_pre4.append(SW_pre)
        if len(SW_pre4) > 4:
            SW_pre4.pop(0)
        lambda_o = lambda_o_array(gv.gama0, gv.gamax, [X_1o, X_2o, X_3o, X_4o], gv.gama2, X_2o, gv.gamaw, SW_pre4, gv.gamay, SY_pre,
                                  o_x, o_y)
        flag_ix = np.where(1 < lambda_o / lambda_max)
        if flag_ix[0].size > 0:
            # print(flag_ix, lambda_o)
            raise ValueError("lambdx_max is too small")
        ix = np.where(np.random.uniform(low=0, high=1, size=NP) <= lambda_o / lambda_max)
        if ix[0].size == 0:
            ix = [0]
        o_x = o_x[ix]
        o_y = o_y[ix]
        data_stamp.append([o_x, o_y])
        SY_pre = [o_x, o_y]

        # generate wt and lambda
        if time_step + 1 in intervention_time:
            NP_next = np.random.poisson(lam=lambda_max)
            wx_next = np.random.uniform(low=0, high=1, size=NP_next)
            wx_next = np.around(np.floor(wx_next * 10) / 10 + 0.05, 2)
            wy_next = np.random.uniform(low=0, high=1, size=NP_next)
            wy_next = np.around(np.floor(wy_next * 10) / 10 + 0.05, 2)
            X_1w_next = get_X1_array(gv, wx_next, wy_next)
            X_2w_next = get_X2_array(gv, wx_next, wy_next)
            X_3w_next = get_X3_array(X3_relative_points, wx_next, wy_next)
            X_4w_next = get_X4_array(X4_relative_points, wx_next, wy_next)
            # intervention_lambda for the whole map
            if with_lambda:
                lambda_x = np.arange(0, 1, 0.1)
                lambda_y = np.arange(0, 1, 0.1)
                lambda_x, lambda_y = np.meshgrid(lambda_x, lambda_y)
                lambda_x = lambda_x.reshape(100)
                lambda_y = lambda_y.reshape(100)
                lambda_X_1w_next = get_X1_array(gv, lambda_x, lambda_y)
                lambda_X_2w_next = get_X2_array(gv, lambda_x, lambda_y)
                lambda_X_3w_next = get_X3_array(X3_relative_points, lambda_x, lambda_y)
                lambda_X_4w_next = get_X4_array(X4_relative_points, lambda_x, lambda_y)
                lambda_next = lambda_w_array(gv.alpha0, gv.alphax,
                                             [lambda_X_1w_next, lambda_X_2w_next, lambda_X_3w_next, lambda_X_4w_next],
                                             gv.alphaw, SW_pre, gv.alphay, SY_pre, lambda_x, lambda_y)
                lambda_next = intervention_C * np.log(lambda_next)
                lambda_next = np.maximum(lambda_next, 0)
                # Each point represents an area of 0.01, and integral is the same as mean
                this_lambda = np.mean(lambda_next)
                lambda_res[f"{time_step + 1}"] = this_lambda
            # non-homogeneous lambda for sampling points
            lambda_w_next = lambda_w_array(gv.alpha0, gv.alphax, [X_1w_next, X_2w_next, X_3w_next, X_4w_next], gv.alphaw, SW_pre,
                                           gv.alphay, SY_pre, wx_next, wy_next)
            lambda_w_next = intervention_C * np.log(lambda_w_next)
            lambda_w_next = np.maximum(lambda_w_next, 0)
            flag_ix_next = np.where(1 < lambda_w_next / lambda_max)
            if flag_ix_next[0].size > 0:
                # print(flag_ix_next, lambda_w_next)
                raise ValueError("lambdx_max is too small")
            ix_next = np.where(np.random.uniform(low=0, high=1, size=NP_next) <= lambda_w_next / lambda_max)
            if ix_next[0].size == 0:
                ix_next = [0]
            wx_next = wx_next[ix_next]
            wy_next = wy_next[ix_next]
            wt_res[f"{time_step + 1}"] = [wx_next, wy_next]

        # next loop
        time_step += 1
    return lambda_res, wt_res



class Data_generator(object):
    def __init__(self, T, N, C, M, base, gv):
        """
        :param T: total time
        :param N: number of simulation
        :param C: intervention_param
        :param M: intervention_length
        :param base: file_base
        """
        self.base = base
        self.T = T
        self.N = N
        self.C = C
        self.M = M
        self.gv = gv

    # observed
    def generating_observation(self):
        SEEDS = list(range(self.N))
        raw_data_list = []
        raw_data_dict = {}
        raw_data_list_path = f"N{self.N}_T{self.T}_O_list.pkl"
        raw_data_dict_path = f"N{self.N}_T{self.T}_O_dict.pkl"
        if os.path.exists(self.base + raw_data_dict_path) and os.path.exists(self.base + raw_data_list_path):
            print(f"{raw_data_dict_path} already exists!")
            print(f"{raw_data_list_path} already exists!")
        else:
            progress = tqdm(total=self.N, desc='Generating observation N: ')
            for seed in SEEDS:
                # empty intervention time list
                raw_data, _ = data_generate_func(gv=self.gv, intervention_time=[], seed=seed, max_time=self.T,
                                                 intervention_C=self.C)
                raw_data_list.append(raw_data)
                data_key = f"N{self.N}_T{self.T}_S{seed}_O"
                raw_data_dict[data_key] = raw_data
                progress.set_description(
                    f'Generating observation N: {seed} | '
                )
                progress.update(1)
            with open(self.base + raw_data_list_path, "wb") as raw_data_list_f:
                pickle.dump(raw_data_list, raw_data_list_f)
            with open(self.base + raw_data_dict_path, "wb") as raw_data_dict_f:
                pickle.dump(raw_data_dict, raw_data_dict_f)
            return raw_data_list

    def generating_intervention(self):
        SEEDS = list(range(self.N))
        intervention_data_dict = {}
        intervention_data_dict_path = f"N{self.N}_T{self.T}_M{self.M}_C{self.C}_I.pkl"
        if os.path.exists(self.base + intervention_data_dict_path):
            print(f"{intervention_data_dict_path} already exists!")
        else:
            progress = tqdm(total=self.N * (self.T - self.M - 1), desc='Generating intervention N: | it: ')
            for seed in SEEDS:
                for it in range(2, self.T - self.M):
                    intervention_time = list(range(it, it + self.M))
                    intervention_data, _ = data_generate_func(gv=self.gv, intervention_time=intervention_time,
                                                              seed=seed, max_time=self.T, intervention_C=self.C)
                    data_key = f"N{self.N}_T{self.T}_S{seed}_M{self.M}_It{it}_C{self.C}_I"
                    intervention_data_dict[data_key] = intervention_data
                    progress.set_description(
                        f'Generating intervention N: {seed} | '
                        f'it: {it} | '
                    )
                    progress.update(1)
            with open(self.base + intervention_data_dict_path, "wb") as intervention_data_dict_f:
                pickle.dump(intervention_data_dict, intervention_data_dict_f)

    def generating_intervention_with_lambda(self):
        SEEDS = list(range(self.N))
        intervention_data_dict = {}
        intervention_data_dict_path = f"N{self.N}_T{self.T}_M{self.M}_C{self.C}_I_lambda.pkl"
        if os.path.exists(self.base + intervention_data_dict_path):
            print(f"{intervention_data_dict_path} already exists!")
        else:
            progress = tqdm(total=self.N * (self.T - self.M), desc='Generating intervention with lambda N: | it: ')
            for seed in SEEDS:
                for it in range(2, self.T - self.M + 2):
                    intervention_time = list(range(it, it + self.M))
                    intervention_data, lambda_res = data_generate_func(gv=self.gv, intervention_time=intervention_time,
                                                                       seed=seed, max_time=self.T,
                                                                       intervention_C=self.C,
                                                                       with_lambda=True)
                    data_key = f"N{self.N}_T{self.T}_S{seed}_M{self.M}_It{it}_C{self.C}_I"
                    intervention_data_dict[data_key + "_lambda"] = lambda_res
                    intervention_data_dict[data_key] = intervention_data
                    progress.set_description(
                        f'Generating intervention with lambda N: {seed} | '
                        f'it: {it} | '
                    )
                    progress.update(1)
            with open(self.base + intervention_data_dict_path, "wb") as intervention_data_dict_f:
                pickle.dump(intervention_data_dict, intervention_data_dict_f)

    def generating_intervention_wt_lambda(self):
        SEEDS = list(range(self.N))
        intervention_data_dict = {}
        intervention_data_dict_path = f"N{self.N}_T{self.T}_M{self.M}_C{self.C}_I_wt_lambda.pkl"
        if os.path.exists(self.base + intervention_data_dict_path):
            print(f"{intervention_data_dict_path} already exists!")
        else:
            progress = tqdm(total=self.N * (self.T - self.M), desc='Generating intervention with lambda N: | it: ')
            for seed in SEEDS:
                for it in range(2, self.T - self.M + 2):
                    intervention_time = list(range(it, it + self.M))
                    lambda_res, wt_res = intervention_wt_lambda_func(gv=self.gv,
                                                                               intervention_time=intervention_time,
                                                                               seed=seed, max_time=self.T,
                                                                               intervention_C=self.C,
                                                                               with_lambda=True)
                    data_key = f"N{self.N}_T{self.T}_S{seed}_M{self.M}_It{it}_C{self.C}_"
                    intervention_data_dict[data_key + "_lambda"] = lambda_res
                    intervention_data_dict[data_key + "wt"] = wt_res
                    progress.set_description(
                        f'generating_intervention_wt_lambda N: {seed} | '
                        f'it: {it} | '
                    )
                    progress.update(1)
            with open(self.base + intervention_data_dict_path, "wb") as intervention_data_dict_f:
                pickle.dump(intervention_data_dict, intervention_data_dict_f)

    def dspp_data(self):
        N = self.N
        T = self.T
        file_path = self.base + f"N{N}_T{T}_O_list.pkl"
        file_name = f"N{N}_T{T}_O_list"
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        all_outcome_list = []
        all_outcome_count_list = []
        for seed_data in data:
            for stamp in seed_data:
                Y, _ = position_encode(stamp[3], n=10, m=10)
                all_outcome_list.append(Y)
                all_outcome_count_list.append(_)
        all_outcome_list = np.array(all_outcome_list)
        np.save(self.base + f"DSPP_{file_name}_data_outcome.npz", all_outcome_list)
        all_outcome_count_list = np.array(all_outcome_count_list)
        np.save(self.base + f"DSPP_{file_name}_data_outcome_count.npz", all_outcome_count_list)

    def ej_data(self):
        n, m = 10, 10
        window = 5
        f = open(self.base + f"N{self.N}_T{self.T}_O_list.pkl", "rb")
        file_name = f"N{self.N}_T{self.T}_O_list"
        raw_data_list = pickle.load(f)
        f.close()

        X_data, label_data, data_xy = [[] for i in range(3)]

        W_pre = [0 for i in range(100)]
        Y_pre = [0 for i in range(100)]
        for raw_data in raw_data_list:
            for stamp in raw_data:
                x3 = stamp[0].reshape(n * m)
                x4 = stamp[1].reshape(n * m)
                W, _ = position_encode(stamp[2], n=10, m=10)
                Y, _ = position_encode(stamp[3], n=10, m=10)
                data_xy.append([W, Y, x3, x4])
            for index in range(self.T - window):
                x = []
                for window_step in range(window):
                    W, Y, x3, x4, = data_xy[index + window_step]
                    x += [W_pre, Y_pre, x3, x4]
                    W_pre = W
                    Y_pre = Y
                X_data.append(x)
                W_ = np.array(data_xy[index + window][0])
                label_data.append(np.sum(W_))
        X1, X2 = mesh_X1_X2(self.gv, n, m)
        X12 = np.array([X1, X2])
        X12 = np.reshape(X12, (1, 2, 100))
        X12 = np.tile(X12, (self.N * (self.T-window), 1, 1))
        X_data = np.array(X_data)
        X_data = np.concatenate((X_data, X12), axis=1)
        np.save(self.base + f"EJ_P_trans{file_name}_data_x.npz", X_data)
        label_data = np.array(label_data)
        np.save(self.base + f"EJ_P_trans{file_name}_data_label.npz", label_data)

@deprecated(reason="no use")
def generating_X1_X2(base):
    """
    map value of static confounder X1 and X2
    :param base:  file_path
    :return: none
    """
    static_X1_path = f"static_X1.pkl"
    static_X2_path = f"static_X2.pkl"
    if os.path.exists(base + static_X1_path) and os.path.exists(base + static_X2_path):
        print("X1, X2 static file already exists!")
    else:
        static_X1, static_X2 = mesh_X1_X2(10, 10)
        with open(base + static_X1_path, "wb") as X1_f:
            pickle.dump(static_X1, X1_f)
        with open(base + static_X2_path, "wb") as X2_f:
            pickle.dump(static_X2, X2_f)

