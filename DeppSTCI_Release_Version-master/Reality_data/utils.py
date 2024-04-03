import numpy
from osgeo import gdal
import numpy as np
import pickle
import os
import pandas as pd

def TIF2coord():
    dataset = gdal.Open("Hansen_GFC-2022-v1.10_lossyear_10N_080W.tif")
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    year_coord = []
    for year in range(1, 23):
        indices = np.where(im_data == year)
        coord_x = im_geotrans[0] + indices[0] * im_geotrans[1] + 0
        coord_x = list(coord_x)
        coord_y = im_geotrans[3] + 0 + indices[1] * im_geotrans[5]
        coord_y = list(coord_y)
        year_coord.append([coord_x, coord_y])
        print(year)
    if os.path.exists("year_loss_coord.pkl"):
        pass
    else:
        with open("year_loss_coord.pkl", "wb") as ylf:
            pickle.dump(year_coord, ylf)


def coord2indexs(data_x, data_y, n=10, m=10):
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    x_step = 10/n
    y_step = 10/m
    data_x = np.floor((data_x + 80) / x_step)
    data_y = np.floor(data_y / y_step)
    return data_x, data_y


def Conflic_year():
    data = pd.read_excel("col_conflict.xlsx")
    data = data.values.tolist()
    Conflic_year = {}
    for year in range(2001, 2023):
        Conflic_year[str(year)] = []
    for data_stamp in data:
        Conflic_year[str(int(data_stamp[0]))].append([data_stamp[2], data_stamp[1]])
    with open("Conlict_year.pkl", "wb") as cff:
        pickle.dump(Conflic_year, cff)
    print()


def year_loss_indexs_count():
    with open("year_loss_coord.pkl", "rb") as ylf:
        data = pickle.load(ylf)
    year_loss_index = []
    for year_data in data:
        data_x, data_y = coord2indexs(year_data[0], year_data[1])
        # 打成格子
        year_indexs = [[0 for i in range(10)] for i in range(10)]
        for point_i in range(len(data_x)):
            x_i = int(data_x[point_i])
            y_i = int(data_y[point_i])
            try:
                year_indexs[x_i][y_i] += 1
            except:
                print(x_i, y_i)
        year_loss_index.append(year_indexs)
    with open("year_loss_index_count.pkl", "wb") as ylcf:
        pickle.dump(year_loss_index, ylcf)


def year_loss_detail_find():
    with open("year_loss_index_count.pkl", "rb") as ylf:
        data = pickle.load(ylf)
    data = np.array(data)
    threshold = numpy.percentile(data, 80)
    data = np.where(data < threshold, 0, 1)
    with open("year_loss_index_01.pkl", "wb") as ylif:
        pickle.dump(data, ylif)
    print(threshold)


def conflict_year_index_count():
    with open("Conlict_year.pkl", "rb") as cff:
        conflict_year = pickle.load(cff)
    conflict_year_count = []
    for year in range(2001, 2023):
        year_index = [[0 for i in range(10)] for i in range(10)]
        year_data = conflict_year[str(year)]
        for data_slice in year_data:
            data_x, data_y = coord2indexs(data_slice[0], data_slice[1])
            x_i = int(data_x)
            y_i = int(data_y)
            year_index[x_i][y_i] += 1
        conflict_year_count.append(year_index)
    conflict_year_count = np.array(conflict_year_count)
    threshold = numpy.percentile(conflict_year_count, 80)
    conflict_year_index = np.where(conflict_year_count < threshold, 0, 1)
    with open("conflict_year_index_01.pkl", "wb") as clif:
        pickle.dump(conflict_year_index, clif)
    print(threshold)

if __name__ == '__main__':
    conflict_year_index_count()

