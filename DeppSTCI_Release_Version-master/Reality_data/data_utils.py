import math
import numpy as np
from PIL import Image
import cv2 as cv
import statistics
from deprecated import deprecated


@deprecated(reason="replaced by geom")
def __line_magnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude


@deprecated(reason="replaced by geom")
def __point_to_line_distance(point, line):
    px, py = point
    x1, y1, x2, y2 = line
    line_magnitude = __line_magnitude(x1, y1, x2, y2)
    distance = -1
    if line_magnitude < 0.00000001:
        print("line points are too closed")
        return distance
    else:
        u1 = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1))
        u = u1 / (line_magnitude * line_magnitude)
        if (u < 0) or (u > 1):
            ix = __line_magnitude(px, py, x1, y1)
            iy = __line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance = iy
            else:
                distance = ix
        else:
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = __line_magnitude(px, py, ix, iy)
    return distance


@deprecated(reason="no use")
def reshape_the_result(data):
    res = []
    for piece in data:
        time_step = piece[0]
        X = piece[1]
        Y = piece[2]
        X1 = piece[3]
        X2 = piece[4]
        X3 = piece[5]
        X4 = piece[6]
        for x, y, x1, x2, x3, x4 in zip(X, Y, X1, X2, X3, X4):
            res.append({
                "time_point": time_step,
                "X": x,
                "Y": y,
                "X1": x1,
                "X2": x2,
                "X3": x3,
                "X4": x4
            })
    return res


def ArrayToPicture(data, filename):
    """
    Draw the grayscale map of data matrix
    :param data:
    :param filename:
    :return:
    """
    data = normalization(data)
    data = 255 * data
    # cv.imshow(filename, data)
    # cv.waitKey()
    cv.imwrite(filename, data)


def PictureToArray(filename):
    """
    gray image to data
    :param filename:
    :return:
    """
    image = Image.open(filename)
    matrix = np.asarray(image)
    return matrix


def normalization(data):
    _range = np.max(abs(data))
    return data / _range


@deprecated(reason="no use")
def point_mid(x_list, y_list):

    x_mid = statistics.mean(x_list)
    y_mid = statistics.mean(y_list)
    return [x_mid, y_mid]


def position_encode(points, n=10, m=10):
    """
    convert coordinate points to 0/1 graph
    :param points: points with x,y coord
    :param n: graph resolution for x
    :param m: graph resolution for y
    :return: 0/1 graph
    """
    res = [0 for i in range(n * m)]
    res_count = [0 for i in range(n * m)]
    for point_x, point_y in zip(points[0], points[1]):
        point_index = int(point_x * 10) + int(point_y * 10) * 10
        res[point_index] = 1
        res_count[point_index] += 1
    return res, res_count
