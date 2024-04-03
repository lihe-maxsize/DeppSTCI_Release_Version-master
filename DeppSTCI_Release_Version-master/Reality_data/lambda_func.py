from confounder_generator import get_min_distance
import numpy as np


def lambda_w_array(alpha0, alphax, X, alphaw, SW_pre, alphay, SY_pre, w_x, w_y):
    """
    batch process for λw generating
    :param alpha0:
    :param alphax:
    :param X: confounder-[X1, X2, X3, X4]
    :param alphaw:
    :param SW_pre:
    :param alphay:
    :param SY_pre:
    :param w_x: x-coord for w
    :param w_y: y-coord for w
    :return:
    e^(α0 + αX + αw * e^(-2 * min_dist_SW) + ay * e^(-2 * min_dist_SY))
    """
    X_1, X_2, X_3, X_4 = X
    res = []
    for xt, yt, x1, x2, x3, x4 in zip(w_x, w_y, X_1, X_2, X_3, X_4):
        w_pre = np.exp(-2 * get_min_distance(SW_pre, xt, yt))
        y_pre = np.exp(-2 * get_min_distance(SY_pre, xt, yt))
        res.append(np.exp(
            alpha0 + alphax[0] * x1 + alphax[1] * x2 + alphax[2] * x3 + alphax[3] * x4 + alphaw * w_pre + alphay * y_pre
        ))
    return np.array(res)


def lambda_o_array(gama0, gamax, X, gama2, X_2o_pre, gamaw, SW_pre4, gamay, SY_pre, o_x, o_y):
    """
    batch process for λy generating
    :param gama0:
    :param gamax:
    :param X: confounder-[X1, X2, X3, X4]
    :param gama2:
    :param X_2o_pre:
    :param gamaw:
    :param SW_pre4:
    :param gamay:
    :param SY_pre:
    :param o_x: x-coord for y
    :param o_y: y-coord for y
    :return:
    e^(γ0 + γX + γ2 * X2 + γw * e^(-2 * min_dist_SW4) + γy * e^(-2 * min_dist_SY))
    """
    X_1, X_2, X_3, X_4 = X
    res = []
    SW_pre4_points_x = []
    SW_pre4_points_y = []
    for point_list in SW_pre4:
        SW_pre4_points_x = SW_pre4_points_x + list(point_list[0])
        SW_pre4_points_y = SW_pre4_points_y + list(point_list[1])
    SW_pre4_points = [SW_pre4_points_x, SW_pre4_points_y]
    for xt, yt, x1, x2, x3, x4, x_2o_pre in zip(o_x, o_y, X_1, X_2, X_3, X_4, X_2o_pre):
        w_pre4 = np.exp(-2 * get_min_distance(SW_pre4_points, xt, yt))
        y_pre = np.exp(-2 * get_min_distance(SY_pre, xt, yt))
        res.append(np.exp(
            gama0 + gamax[0] * x1 + gamax[1] * x2 + gamax[2] * x3 + gamax[3] * x4
            + gama2 * x_2o_pre + gamaw * w_pre4 + gamay * y_pre
        ))
    return np.array(res)
