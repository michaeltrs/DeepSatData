import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from pyproj import Proj, transform
import re 


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


def get_points_from_str_poly(str_poly):
    return np.array([[float(j) for j in i.split(" ") if j != ''] for i in str_poly.split("(")[-1].split(")")[0].split(",")])


# def find_samples_in_poly(N, W, h, w, T, data_df):
#     is_in_prod = (N >= data_df['north']) & (N - 10 * h <= (data_df['north'])) & \
#                  (W <= data_df['west']) & (W + 10 * w >= (data_df['west']))
#     prod_doy = get_doy(T)
#     data_doy = (data_df[[c for c in data_df.columns if c.startswith("doy")]] * 365.0001).round(0).astype(np.int32)
#     doy_idx = (data_doy == prod_doy).any(axis=1)
#     return data_df[is_in_prod & doy_idx]
