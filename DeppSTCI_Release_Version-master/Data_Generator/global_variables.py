import shapely.geometry as geom

SW0 = [[0., 1.], [0., 1.]]
SY0 = [[0., 1.], [0., 1.]]
rou0_x3 = -0.2  # tv
rou1_x3 = 2.3  # tv
rou0_x4 = -0.2
rou1_x4 = 2.8
alpha0 = -1.
alphax = [1., 1., 1., 1.]
alphaw = 1.
alphay = 1.
gama0 = -1.
gamax = [1., 1., 1., 1.]
gama2 = 1.
gamaw = 1.
gamay = 1.

# line1 = 2.5x - 1.25 x~(0.5-0.8) (0.5, 0) (0.8, 1)
line11 = [0.5, 0., 0.8, 1.]
line1 = geom.LineString([(0.5, 0.), (0.8, 1.)])
line22 = [0, 0.5, 0.5, 0.8]
# line2 = 0.6x + 0.5  x~(0-0.5)   (0, 0.5) (0.5, 0.8)
line2 = geom.LineString([(0, 0.5), (0.5, 0.8)])

# arc
arc1 = geom.LineString(
    [(0.5, 1), (0.6, 0.7), (0.7, 0.6), (0.8, 0.5417), (0.9, 0.5101), (1.0, 0.5)])
arc2 = geom.LineString([(0, 0.5), (0.1, 0.4899), (0.2, 0.4583), (0.3, 0.4), (0.4, 0.3), (0.5, 0.1)])
