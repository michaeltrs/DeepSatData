import matplotlib.pyplot as plt
import numpy as np
from shapely import geometry
from shapely.geometry import Polygon
from pyproj import Proj, transform
import re 
from simplification.cutil import simplify_coords
from sentinelsat import geojson_to_wkt


class GeoTransform:
    def __init__(self, intr, outtr, loc2loc=False):
        """
        - loc2loc: from local to local coord system. In this case tranform remains x, y -> x, y,
                   otherwise x, y -> y, x
        """
        intr = str(intr)
        outtr = str(outtr)
        if not intr.isnumeric(): intr = get_epsg_code(intr)
        if not outtr.isnumeric(): outtr = get_epsg_code(outtr)
        self.inProj = Proj(init='epsg:%s' % intr)  # %d' % get_epsg_code(country))
        self.outProj = Proj(init='epsg:%s' % outtr)  # 2154')
        self.loc2loc = loc2loc

    def __call__(self, x, y):
        yout, xout = transform(self.inProj, self.outProj, x, y)
        if self.loc2loc:
            return yout, xout
        return xout, yout


def make_AOI(coords, transform):
    if type(coords) == str:
        x = [float(x) for x in re.findall("[+-]?\d+(?:\.\d+)?", coords)]
        x = np.array(x).reshape(-1, 2)

        points = []
        for point in x:
            points.append(transform(point[0], point[1]))
        points.append(transform(x[0][0], x[0][1]))

        poly = make_poly(points[:-1])
        footprint = coords
        AOI = Polygon(points)

    elif type(coords) in [list, tuple]:
        if len(coords) == 2:  # assume NW, SE boxx coords
            NW, SE = coords
            NW_glob = transform(NW[1], NW[0])
            SE_glob = transform(SE[1], SE[0])

            poly = make_rect_poly(NW_glob, SE_glob)
            footprint = geojson_to_wkt(poly)
            AOI = Polygon([[NW_glob[1], NW_glob[0]],
                           [NW_glob[1], SE_glob[0]],
                           [SE_glob[1], SE_glob[0]],
                           [SE_glob[1], NW_glob[0]],
                           [NW_glob[1], NW_glob[0]]])

        else:
            points = []
            for point in coords:
                points.append(transform(point[0], point[1]))
            points.append(transform(x[0][0], x[0][1]))

            poly = make_poly(points[:-1])
            footprint = geojson_to_wkt(poly)
            AOI = Polygon(points)
    
    return poly, footprint, AOI


def make_poly(points, ret_points=False):
    points.append(points[0])
    poly = {"type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": {
                "type": "Polygon",
                "coordinates": [[points]]} }]}
    if ret_points:
        return poly['features'][0]['geometry']['coordinates']
    return poly


def make_rect_poly(nw, se, ret_points=False):
    # W, N
    poly = {"type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": {
                "type": "Polygon",
                "coordinates": [[[nw[1], nw[0]],
                                 [nw[1], se[0]],
                                 [se[1], se[0]],
                                 [se[1], nw[0]],
                                 [nw[1], nw[0]]]]} }]}
    if ret_points:
        return poly['features'][0]['geometry']['coordinates']
    return poly


def get_epsg_code(country):
    epsg_code = {'germany': 32632, 'senegal': 32628, 'france': 32631}
    return epsg_code[country]


def plot_poly(points, c=None, newfig=True):
    if type(points) in [list, tuple]:
        points = np.array(points)
    if c is None:
        c = "r"
    if newfig:
        plt.figure()
    for i in range(points.shape[0] - 1):
        plt.plot(points[i:i + 2, 0], points[i:i + 2, 1], c=c)
        plt.scatter(points[i, 0], points[i, 1], c=c)


def get_points_from_str_poly(str_poly):
    return np.array([[float(j) for j in i.split(" ") if j != ''] for i in str_poly.split("(")[-1].split(")")[0].split(",")])


# eometry
def get_line_eq(p1, p2, h=1e-7):
    '''
    P: (x, y)
    '''
    denom = p2[0] - p1[0]
    if denom == 0:
        denom = h
    a = (p2[1] - p1[1]) / denom
    b = (p1[1] * p2[0] - p2[1] * p1[0]) / denom
    return a, b


def get_perp_line(p1, p2, p3, h=1e-7):
    a, b = get_line_eq(p1, p2)
    if a == 0:
        a = h
    a_ = - 1. / a
    b_ = p3[1] + 1. / a * p3[0]
    return a_, b_


def get_perp_bisect(p1, p2, p3, h=1e-7):
    '''
    p1, p2: line segment end points
    p3: point outside line
    '''
    a, b = get_line_eq(p1, p2)
    if a == 0:
        a = h
    a_ = - 1. / a
    b_ = p3[1] + 1. / a * p3[0]
    x = (b_ - b) / (a - a_)
    y = a * x + b
    return x, y


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def is_between(p1, p2, p3):
    xmax = max((p1[0]), p2[0])
    xmin = min((p1[0]), p2[0])
    if (p3[0] < xmax) & (p3[0] > xmin):
        return True
    else:
        return False


def min_dist(p1, p2, p3):
    p4 = get_perp_bisect(p1, p2, p3)
    if is_between(p1, p2, p4):
        return p4, dist(p3, p4)
    d1 = dist(p3, p1)
    d2 = dist(p3, p2)
    return [p1, p2][np.argmin([d1, d2])], min((d1, d2))


def closest_point_to_poly(poly, point, return_dist=False):
    D = []
    P = []
    for i in range(len(poly)):
        if i == (len(poly) - 1):
            p, d = min_dist(poly[i], poly[0], point)
        else:
            p, d = min_dist(poly[i], poly[i + 1], point)
        D.append(d)
        P.append(p)
    idx = np.argmin(D)
    if return_dist:
        return D[idx]
    return P[idx]


def distance_pix_to_poly(poly, point):
    poly_point = closest_point_to_poly(poly, point)
    return dist(poly_point, point)



def add_points(poly, numpoints=100):
    # increase number of points by splitting largest line segments in half
    while poly.shape[0] < numpoints:
        idx = np.argmax([dist(poly[i], poly[i+1]) for i in range(poly.shape[0]-1)])
        new_point = (poly[idx] + poly[idx+1]) / 2.
        poly = np.insert(poly, idx+1, new_point, 0)
    return poly


def interp1d(N, Nmax, Nmin, tmax, tmin):
    return (N - Nmin) / (Nmax - Nmin) * (tmax - tmin) + tmin


def simplify_poly_points(poly, numpoints=20, iter_max=20):
    numpoints_init = poly.shape[0]

    if numpoints_init == numpoints:
        return poly
    elif numpoints_init < numpoints:
        return add_points(poly, numpoints)
    else:  # get initial values of t that lead to larger and smaller polygons
        Nmax = numpoints_init
        t = 5
        while simplify_coords(poly, t).shape[0] >= numpoints:
            t *= 2
        Nmin = simplify_coords(poly, t).shape[0]
        Tmax, Tmin = 0, t

    iter = 0
    while True:
        t = interp1d(numpoints, Nmax, Nmin, Tmax, Tmin)
        poly_ = simplify_coords(poly, t)
        N = poly_.shape[0]
        # print(N, t)
        if N == numpoints:
            break
        elif N > numpoints:
            Nmax, Tmax = N, t
        elif N < numpoints:
            Nmin, Tmin = N, t
        iter += 1
        if iter > iter_max:
            poly_ = simplify_coords(poly, Tmin)
            return add_points(poly_, numpoints)
    return poly_


def is_valid(parcel_poly, pxmin, pymax, res=10):
    """
    checks if parcel_poly polygon has valid shape
    """
    isvalid = True
    i = 0
    j = 0
    pix_points = [[pxmin + loc[0] * res, pymax - loc[1] * res] for loc in
                  [[j, i], [j + 1, i], [j + 1, i + 1], [j, i + 1], [j, i]]]
    try:
        parcel_poly.intersection(geometry.Polygon(pix_points)).area
    except:
        isvalid = False
    return isvalid


def str_line_eq(points, h=1e-1):
    assert points.shape == (2, 2), 'Two points must be used to derive straight line equation'
    x1, y1 = points[0]
    x2, y2 = points[1]
    denom = x2 - x1
    if denom == 0:
        denom = h
    a = (y2 - y1) / denom  # (x2 - x1)
    b = (y1 * x2 - x1 * y2) / denom  # (x2 - x1)
    return a, b


# def find_samples_in_poly(N, W, h, w, T, data_df):
#     is_in_prod = (N >= data_df['north']) & (N - 10 * h <= (data_df['north'])) & \
#                  (W <= data_df['west']) & (W + 10 * w >= (data_df['west']))
#     prod_doy = get_doy(T)
#     data_doy = (data_df[[c for c in data_df.columns if c.startswith("doy")]] * 365.0001).round(0).astype(np.int32)
#     doy_idx = (data_doy == prod_doy).any(axis=1)
#     return data_df[is_in_prod & doy_idx]
