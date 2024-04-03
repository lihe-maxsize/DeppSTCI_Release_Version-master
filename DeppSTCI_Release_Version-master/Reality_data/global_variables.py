import shapely.geometry as geom
import pickle
import numpy as np


class global_variable(object):
    def __init__(self, rou0_x3=-0.2, rou1_x3=2.3, rou0_x4=-0.2, rou1_x4=2.8, alpha0=-1., alphax=None,
                 alphaw=1., alphay=1., gama0=1., gamax=None, gama2=1., gamaw=1., gamay=1.):
        if gamax is None:
            gamax = [1., 1., 1., 1.]
        if alphax is None:
            alphax = [1., 1., 1., 1.]
        with open("year_loss_index_01.pkl", "rb") as ylif:
            yli_data = pickle.load(ylif)
        with open("conflict_year_index_01.pkl", "rb") as cfif:
            cfi_data = pickle.load(cfif)
        yli_2002 = yli_data[1]  # fetch 2002
        cfi_2002 = cfi_data[1]
        yli_2002_coord = np.where(yli_2002==1)
        cfi_2002_coord = np.where(cfi_2002==1)
        self.SY0 = [yli_2002_coord[0]/10, yli_2002_coord[1]/10]
        self.SW0 = [cfi_2002_coord[0]/10, cfi_2002_coord[1]/10]
        self.rou0_x3 = rou0_x3  # tv
        self.rou1_x3 = rou1_x3  # tv
        self.rou0_x4 = rou0_x4
        self.rou1_x4 = rou1_x4
        self.alpha0 = alpha0
        self.alphax = alphax
        self.alphaw = alphaw
        self.alphay = alphay
        self.gama0 = gama0
        self.gamax = gamax
        self. gama2 = gama2
        self.gamaw = gamaw
        self.gamay = gamay
        self.line1 = geom.LineString([
            (0.23389127, 0.7783898),
            (0.35715316, 0.30922253),
            (0.39394775, 0.42936427)
        ])

        self.line2 = geom.LineString([
            (0.39394775, 0.42936427),
            (0.59631802, 0.46629492),
            (0.69566342, 0.70579619),
            (0.76189369, 0.7755867),
        ])

        self.line3 = geom.LineString([
            (0.39762721, 0.43121125),
            (0.44546018, 0.61564696),
            (0.6883045, 0.70947212),
            (0.57792072, 0.109713926),
            (0.77661152, 0.113347569)
        ])

        self.line4 = geom.LineString([
            (0.44729991, 0.61564696),
            (0.53008775, 0.109350311)
        ])

        self.line5 = geom.LineString([
            (0.44706994, 0.62278205),
            (0.32265823, 0.84228317)
        ])
        # arc
        self.arc1 = geom.LineString([
            (0.847053, 1.218998),
            (0.820686, 0.681763),
            (0.1194221, 0.638109),
            (0.1211799, 0.261412),
            (0.1027229, 0.156015),
            (0.943733, -0.22184),
            (0.113166, 0.129656),
            (0.236213, 0.742812),
            (0.504279, 1.107088)
        ])
