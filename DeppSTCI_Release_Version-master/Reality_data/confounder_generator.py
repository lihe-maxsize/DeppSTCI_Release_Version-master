import numpy as np
# from global_variables import *
import math
from data_utils import *
import shapely.geometry as geom


def get_X1(gv, xt, yt):
    """
    given a point, then calculate the X1 value for this point
    :param gv: global variable
    :param xt: x-coord
    :param yt: y-coord
    :return: X1 value
    1.2 e^(-2 * min_linedist)
    """
    point = geom.Point(xt, yt)
    dis1 = point.distance(gv.line1)
    dis2 = point.distance(gv.line2)
    dis3 = point.distance(gv.line3)
    dis4 = point.distance(gv.line4)
    dis5 = point.distance(gv.line5)
    dis = min([dis1, dis2, dis3, dis4, dis5])
    return 1.2 * np.exp(-2 * dis)


def get_X1_array(gv, xt, yt):
    """
    batch process for X1 generating
    :param gv: global variable
    :param xt: np.array type x-coord
    :param yt: np.array type y-coord
    :return: X1 values for points
    1.2 e^(-2 * min_linedist)
    """
    res = []
    for xt, yt in zip(xt, yt):
        point = geom.Point(xt, yt)
        dis1 = point.distance(gv.line1)
        dis2 = point.distance(gv.line2)
        dis3 = point.distance(gv.line3)
        dis4 = point.distance(gv.line4)
        dis5 = point.distance(gv.line5)
        dis = min([dis1, dis2, dis3, dis4, dis5])
        res.append(1.2 * np.exp(-2 * dis))
    return np.array(res)


def get_X2(gv, xt, yt) -> float:
    """
    given a point, then calculate the X2 value for this point
    :param gv: global variable
    :param xt: x-coord
    :param yt: y-coord
    :return: X2 value
    e^(-3 * min_arcdist)
    """

    point = geom.Point(xt, yt)
    dis1 = point.distance(gv.arc1)
    return np.exp(-3 * dis1)


def get_X2_array(gv, xt, yt):
    """
    batch process for X2 generating
    :param gv: global variable
    :param xt: np.array type x-coord
    :param yt: np.array type y-coord
    :return: X2 values for points
    e^(-3 * min_arcdist)
    """
    res = []
    for xt, yt in zip(xt, yt):
        point = geom.Point(xt, yt)
        dis1 = point.distance(gv.arc1)
        res.append(np.exp(-3 * dis1))
    return np.array(res)


def get_min_distance(points, x, y):
    """
    min-distance between relative-points and key points
    :param points:  all relative-points
    :param x:  x-coord
    :param y:  y-coord
    :return:  min-distance
    """
    pxs = points[0]
    pys = points[1]
    distances = [math.sqrt((px - x) ** 2 + (py - y) ** 2) for px, py in zip(pxs, pys)]
    min_dis = min(distances)
    return min_dis


def get_X3_array(X3_relative_points, w_x, w_y):
    """
    batch process for X3 generating
    :param X3_relative_points: necessary relative points for X3 calculation
    :param w_x: np.array type x-coord
    :param w_y: np.array type y-coord
    :return: X3 values for points
    1.2 * e^(-2 dis_min)
    """
    res = []
    for xt, yt in zip(w_x, w_y):
        dis = get_min_distance(X3_relative_points, xt, yt)
        res.append(1.2 * np.exp(-2 * dis))
    return np.array(res)


def get_X4_array(X4_relative_points, w_x, w_y):
    """
    batch process for X4 generating
    :param X4_relative_points: necessary relative points for X4 calculation
    :param w_x: np.array type x-coord
    :param w_y: np.array type y-coord
    :return: X4 values for points
    e^(-3 dis_min)
    """
    res = []
    for xt, yt in zip(w_x, w_y):
        dis = get_min_distance(X4_relative_points, xt, yt)
        res.append(np.exp(-3 * dis))
    return np.array(res)


def get_relative_points(rou0, rou1, gv) -> list:
    """
    λ = e^(ρ0 + ρ1 * X1), Non-homogeneous Poisson distribution rejects sampling
    :param rou0: param
    :param rou1: param
    :param gv: global variable
    :return: sampling points
    """
    lambda_x = 150
    Np = np.random.poisson(lam=lambda_x)
    xt = np.random.uniform(size=Np)
    yt = np.random.uniform(size=Np)
    lamda = np.exp(rou0 + rou1 * get_X1_array(gv, xt, yt))
    flag_ix = np.where(1 < lamda / lambda_x)
    if flag_ix[0].size > 0:
        print(flag_ix, lamda)
        raise ValueError("lambdx_x is too small")

    ix = np.where(np.random.uniform(size=Np) < np.exp(rou0 + rou1 * get_X1_array(gv, xt, yt)) / lambda_x)
    if ix[0].size == 0:
        ix = [0]
    xt = xt[ix]
    yt = yt[ix]
    return [xt, yt]


def mesh_image_X1_X2(gv, n=100, m=100):
    """
    Draw the grayscale map of X1 and X2
    :param gv: global variable
    :param n: x_resolution
    :param m: y_resolution
    :return:
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, m)
    X, Y = np.meshgrid(x, y)
    X = np.reshape(X, newshape=(m*n))
    Y = np.reshape(Y, newshape=(m*n))

    # get X1
    X_1 = get_X1_array(gv, X, Y)
    X_1 = np.reshape(X_1, newshape=(n, m))
    ArrayToPicture(X_1, "X_1.jpg")

    # get X2
    X_2 = get_X2_array(gv, X, Y)
    X_2 = np.reshape(X_2, newshape=(n, m))
    ArrayToPicture(X_2, "X_2.jpg")
    print()


def mesh_X1_X2(gv, n=10, m=10):
    """
    Plane value of X1 and X2
    :param gv: global variable
    :param n: x_resolution
    :param m: y_resolution
    :return:
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, m)
    X, Y = np.meshgrid(x, y)

    X = np.reshape(X, newshape=(m*n))
    Y = np.reshape(Y, newshape=(m*n))

    # get X1
    X_1 = get_X1_array(gv, X, Y)
    # get X2
    X_2 = get_X2_array(gv, X, Y)

    return X_1, X_2
