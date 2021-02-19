import matplotlib.pyplot as plt
import numpy as np
from pyproj import Proj, transform


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
