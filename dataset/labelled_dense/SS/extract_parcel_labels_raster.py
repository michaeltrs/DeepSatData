import argparse
import pandas as pd
import numpy as np
from shapely import geometry
import os
from glob import glob
from multiprocessing import Pool
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.insert(0, dir(dir(path[0])))
    __package__ = "examples"
from utils.geospatial_data_utils import GeoTransform, get_points_from_str_poly
from utils.multiprocessing_utils import split_df
from utils.sentinel_products_utils import get_S2prod_info
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy


def is_valid(parcel_poly, pxmin, pymax):
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


def plot_poly(points, c=None, newfig=False):
    if type(points) in [list, tuple]:
        points = np.array(points)
    if c is None:
        c = "r"
    if newfig:
        plt.figure()
    for i in range(points.shape[0] - 1):
        plt.plot(points[i:i + 2, 0], points[i:i + 2, 1], c=c)


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


def extract_parcel_labels_raster(inputs):

    # rank = 0
    rank, geodata, W, N, Wp, Np, year, crs = inputs
    # rank, geodata, W, N, Wp, Np, year, crs = inputs[0]

    # # arrays to save
    # AOI_labels = np.zeros((int(np.round(dy / res)), int(np.round(dx / res))), dtype=np.float32) # + max_label + 1
    # AOI_ids = np.zeros((int(np.round(dy / res)), int(np.round(dx / res))), dtype=np.float32)
    # AOI_masks = AOI_ids.copy()
    # # additional/helper arrays
    # AOI_ratios = AOI_ids.copy()
    year_savedir = os.path.join(savedir, 'Y%s_N%s_W%s_R%d_CRS%s' % (year, N, W, res, crs))
    if not os.path.exists(year_savedir):
        os.makedirs(year_savedir)

    saved_data_info = []
    # invalid_shapes = []
    for ii in range(geodata.shape[0]):
        # ii = 3600  # 4500
        print("process %d, parcel %d of %d" % (rank, ii+1, geodata.shape[0]))
        parcel_poly = geodata['geometry'][ii]
        label = geodata['ground_truth'][ii]
        id = geodata['id'][ii]

        points = get_points_from_str_poly(parcel_poly)
        anchor = np.array(geometry.Polygon(points).centroid)
        # anchor = points.mean(axis=0)
        N0 = anchor[1] + sample_size * res / 2.
        W0 = anchor[0] - sample_size * res / 2.

        # correct for non integer offset wrt product Nmax, Wmax (top-left) coordinates
        dN = (Np - N0) % 60
        dW = (W0 - Wp) % 60
        N0 += dN
        W0 -= dW
        # anchor[1] = N0 - sample_size * res / 2.
        # anchor[0] = W0 + sample_size * res / 2.
        anchor = np.array([W0 + sample_size * res / 2., N0 - sample_size * res / 2.])
        # anchor = points.mean(axis=0) #- sample_size * res / 2

        # pr = points - anchor
        pr = (points - anchor + sample_size * res / 2)  # !!!
        parcel_poly = geometry.Polygon(pr)

        pxmin, pymin = pr.min(axis=0)
        pxmax, pymax = pr.max(axis=0)

        # DONT DO VERY SMALL ONES
        # if ((pxmax - pxmin) < 20) or ((pymax - pymin) < 20):

        if not is_valid(parcel_poly, pxmin, pymax):
            try:
                int_area = sum(
                    [geometry.Polygon(np.array(pol.coords[:])).area for pol in parcel_poly.buffer(0).interiors])
                ext_area = geometry.Polygon(np.array(parcel_poly.buffer(0).exterior.coords[:])).area
                if int_area / ext_area < 0.05:  # threshold for discarding a parcel polygon
                    print("included, number of interior areas %d, intarea: %.0f, extarea: %.0f, ratio: %.4f" %
                          (len(parcel_poly.buffer(0).interiors), int_area, ext_area, int_area / ext_area))
                    parcel_poly = geometry.Polygon(np.array(parcel_poly.buffer(0).exterior.coords[:]))
                    pr = np.stack([np.array(i) for i in parcel_poly.exterior.coords.xy]).T
                    pxmin, pymin = pr.min(axis=0)
                    pxmax, pymax = pr.max(axis=0)
                else:
                    print("excluded, number of interior areas %d, intarea: %.0f, extarea: %.0f, ratio: %.4f" %
                          (len(parcel_poly.buffer(0).interiors), int_area, ext_area, int_area / ext_area))
                    values = geodata.iloc[ii].to_list()
                    for v in [N0, W0, None]:
                        values.append(v)
                    saved_data_info.append(values)
                    continue
            except:
                continue

        # labels = np.zeros((sample_size, sample_size), dtype=np.float32)
        # ids = labels.copy()
        ratios = np.zeros((sample_size, sample_size), dtype=np.float32)
        alpha = ratios.copy()
        # global_alpha = ratios.copy()
        global_beta = ratios.copy()
        # local_alpha = ratios.copy()
        local_beta = ratios.copy()

        row0 = int(np.floor((1 - pymax / (sample_size * res)) * sample_size))
        row1 = int(np.ceil((1 - pymin / (sample_size * res)) * sample_size))
        col0 = int(np.floor(pxmin / (sample_size * res) * sample_size))
        col1 = int(np.ceil(pxmax / (sample_size * res) * sample_size))  # + 1
        # row0 = int((1 - pr[:, 1].max() / dy) * AOI_labels.shape[0])
        # row1 = int((1 - pr[:, 1].min() / dy) * AOI_labels.shape[0])
        # col0 = int(pr[:, 0].min() / dx * AOI_labels.shape[1])
        # col1 = int(pr[:, 0].max() / dx * AOI_labels.shape[1]) + 1

        # H, W = sample_size, sample_size
        Height, Width = row1 - row0, col1 - col0

        # if (Height < 5) or (Width)
        # bl = False

        for i in range(Height):
            # if bl:
            #     break
            for j in range(Width):
                # i, j = 0, 3
                if (row0 + i) * (col0 + j) < 0:
                    continue

                try:

                    pix_points = [[pxmin + loc[0] * res, pymax - loc[1] * res] for loc in
                                  [[j, i], [j + 1, i], [j + 1, i + 1], [j, i + 1], [j, i]]]

                    pix_poly = geometry.Polygon(pix_points)

                    value = parcel_poly.intersection(pix_poly).area / res ** 2
                    if (0 < value) and (value < 1):  # parcel cuts through pixel
                        # print(i, j)
                        # bl = True
                        global_points = np.array(parcel_poly.boundary.intersection(pix_poly.boundary))
                        if global_points.shape[0] > 2: # !!!
                            global_points = global_points[:2]
                        global_params = str_line_eq(global_points)
                        alpha[row0 + i + 1, col0 + j + 1] = global_params[0]
                        # global_alpha[row0 + i + 1, col0 + j + 1] = global_params[0]
                        global_beta[row0 + i + 1, col0 + j + 1] = global_params[1] / (sample_size * res)
                        local_points = (global_points - np.array([pxmin + j * res, pymax - i * res])) / res
                        local_params = str_line_eq(local_points)
                        # local_alpha[row0 + i + 1, col0 + j + 1] = local_params[0]
                        local_beta[row0 + i + 1, col0 + j + 1] = local_params[1]

                        # break

                    if value == 0:  # no intersection
                        continue

                    # labels[row0 + i + 1, col0 + j + 1] = label
                    ratios[row0 + i + 1, col0 + j + 1] = value
                    # ratios[col0 + i + 1, row0 + j + 1] = value

                    # ids[row0 + i + 1, col0 + j + 1] = id

                except:
                    continue
        # replace global, local alpha with alpha
        sample = {'N': N0, 'W': W0, 'boundary': pr / res, 'label': label, 'id': id, 'ratios': ratios,
                  'alpha': alpha, 'global_beta': global_beta, 'local_beta': local_beta}
        impath = os.path.join(year_savedir, 'N%d_E%d_ground_truths.pickle' % (N0, W0))
        with open(impath, 'wb') as handle:
            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

        values = geodata.iloc[ii].to_list()
        for v in [N0, W0, impath]:
            values.append(v)
        saved_data_info.append(values)

    return saved_data_info


    # lines = []
    # boundary = parcel_poly.boundary
    # if boundary.type == 'MultiLineString':
    #     for line in boundary:
    #         lines.append(line)
    # else:
    #     lines.append(boundary)
    #
    # ########################################################################
    #
    #
    #
    # points = pix_poly.boundary.intersection(parcel_poly.boundary)  # multipoint
    # points = np.array(points)
    #
    #
    # print(points)
    # plt.figure()
    # plt.imshow(ratios)
    # plt.scatter((anchor[0]-W0) / res, (N0 - anchor[1])/res)


#     plt.figure()
#     plt.hist(global_alpha[global_alpha != 0], 20)
#
#     plt.figure()
#     plt.hist(alpha[alpha != 0], 20)
#
#     plt.figure()
#     plt.hist(np.tanh(global_beta[global_beta != 0]))
#
#     plt.figure()
#     plt.hist(np.tanh(local_beta[local_beta != 0]))
#
#     plt.figure()
#     plt.hist(np.tanh(global_alpha[global_alpha != 0]), 20)
#
#     plt.figure()
#     plt.hist(np.tanh(global_beta[global_beta > 0] / 1000.))
#
    # plt.figure()
    # plt.imshow(alpha)  # , ::-1])
    # plt.title('alpha')
    # plt.colorbar()
    # pr1 = pr / res # - np.array([2.5, 8])
    # pr1[:, 1] = 100 - pr1[:, 1]
    # pr1 = pr1
    # plot_poly(pr1, newfig=False)
#
#     plt.figure()
#     plt.imshow(global_beta)  # , ::-1])
#     plt.title('global_beta')
#     plt.colorbar()
#     plot_poly(pr1, newfig=False)
#
#
# def dot(x1, x2):
#     return x1.dot(x2)
#
#
# def norm_dot(x1, x2):
#     x1 = x1 / np.linalg.norm(x1)
#     x2 = x2 / np.linalg.norm(x2)
#     return x1.dot(x2)
#
#
# l1 = np.array([-100, 300])
# l2 = np.array([100, -300])
# l3 = np.array([20, -5])
#
# dot(l1, l2)
# dot(l2, l3)
#
# norm_dot(l1, l2)
# norm_dot(l2, l3)




# plot_poly(pix_points, newfig=False)

# x = np.linspace(0, 100, 100)
# y = -0.407134 * x + (100 - 103.457/10)  # 103.457
#
# plt.plot(x, y)

    # # plt.figure()
    # plt.plot(*geometry.Polygon(pr1).exterior.xy)
    # # for i in range(pr1.shape[0] - 1):
    # #     plt.plot(pr1[i:i + 2, 0], pr1[i:i + 2, 1], c='r')
    # plt.scatter(1, 1)

    # return AOI_labels, AOI_ids, AOI_masks, AOI_ratios, pd.DataFrame(invalid_shapes)


def main():
    # ground truth data
    gt_df = pd.read_csv(ground_truths_file)
    if 'id' not in gt_df:
        print('Column "id" not included. Assigning values from 1 to file size')
        gt_df['id'] = range(1, gt_df.shape[0]+1)
    # gt_df['id'] = range(1, gt_df.shape[0]+1)
    assert (gt_df['crs'] == gt_df['crs'].iloc[0]).all(), \
        "Polygons corresponding to multiple CRS were found in %s" % ground_truths_file
    crs = gt_df['crs'].iloc[0]
    yearly_grouped_gt = gt_df.groupby('year')
    years = list(yearly_grouped_gt.groups.keys())
    print("found ground truth data for years %s" % ", ".join([str(i) for i in years]))
    if 0 in gt_df['ground_truth'].drop_duplicates():
        gt_df['ground_truth'] += 1

    # sentinel products
    imdirs = glob("%s/*.SAFE/GRANULE/**/IMG_DATA" % products_dir)
    prod_df = get_S2prod_info(imdirs)
    assert (prod_df['West']==prod_df['West'].iloc[0]).all() and (prod_df['North']==prod_df['North'].iloc[0]).all(),\
    "Sentinel products corresponding to multiple tiles were found in %s" % products_dir
    geotr = GeoTransform(intr=prod_df['crs'].iloc[0].split(':')[1], outtr=gt_df['crs'].iloc[0], loc2loc=gt_df['crs'].iloc[0] != '4326')
    prod_WN = prod_df[['West', 'North']].iloc[0].tolist()
    prod_WN = geotr(prod_WN[0], prod_WN[1])  # in ground truth data coordinate system
    d = (10 * prod_df[['height', 'width']].iloc[0].values).tolist()

    # find all ground truth data that fall inside sentinel product
    prod_poly = geometry.Polygon([[prod_WN[0] + loc[0] * d[0], prod_WN[1] - loc[1] * d[1]] for loc in
                                  [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]])
    print(prod_poly)
    def f(x):
        try:
            x = get_points_from_str_poly(x)
            W = x[:, 0].min()
            E = x[:, 0].max()
            S = x[:, 1].min()
            N = x[:, 1].max()
            x = geometry.Polygon(x)
            inratio = prod_poly.intersection(x).area / x.area
            return np.array([N, S, W, E, inratio])
        except:
            return np.array([0, 0, 0, 0, 0])

    gt_df[['N', 'S', 'W', 'E', 'inratio']] = np.stack(gt_df['geometry'].apply(f).values)
    gt_df = gt_df[gt_df['inratio'] == 1.0]
    print("found %d polygons inside sentinel tile" % gt_df.shape[0])

    N = int(np.ceil(gt_df['N'].max()))   # N-maxy
    # S = int(np.floor(gt_df['S'].min()))  # S-miny
    # E = int(np.ceil(gt_df['E'].max()))   # E-maxx
    W = int(np.floor(gt_df['W'].min()))  # W-minx

    # # increase AOI dimensions to match integer multiple of sample size
    # if np.ceil((maxy - miny) / (sample_size * res)) != (maxy - miny) / (sample_size * res):
    #     dy = (np.ceil((maxy - miny) / (sample_size * res)) - (maxy - miny) / (sample_size * res)) * (sample_size * res)
    #     miny = miny - dy
    # if np.ceil((maxx - minx) / (sample_size * res)) != (maxx - minx) / (sample_size * res):
    #     dx = (np.ceil((maxx - minx) / (sample_size * res)) - (maxx - minx) / (sample_size * res)) * (sample_size * res)
    #     maxx = maxx + dx
    # dx = maxx - minx
    # dy = maxy - miny
    # anchor = minx, miny  # WS

    pool = Pool(num_processes)

    for year in years:
        # year = years[0]

        geodata = gt_df[gt_df['year'] == year].reset_index(drop=True)

        inputs = [[i, df_, W, N, prod_WN[0], prod_WN[1], year, crs] for i, df_ in enumerate(split_df(geodata, num_processes))]

        outputs = pool.map(extract_parcel_labels_raster, inputs)

        saved_data_info = pd.concat(pd.DataFrame(out) for out in outputs)
        save_name = os.path.join(savedir, 'Y%s_N%s_W%s_R%d_CRS%s' % (year, N, W, res, crs), 'saved_data_info.csv')
        saved_data_info.columns = ['id', 'ground_truth', 'crs', 'year', 'geometry', 'Np', 'Sp', 'Wp', 'Ep',
       'inratio', 'Dy', 'Dx', 'D', 'Ntl', 'Wtl', 'filepath']
        saved_data_info.to_csv(save_name, index=False)

        # d = pd.read_csv(save_name)
        # d['filepath'] = d['filepath'].apply(lambda s: os.path.join('/'.join(s.split('/')[:-1]),
        #                                                            'Y2018_N6650384_W799943_R10_CRS2154',
        #                                                            s.split('/')[-1]))
        # AOI_labels = np.stack([out_[0] for out_ in outputs])
        # AOI_ids = np.stack([out_[1] for out_ in outputs])
        # AOI_masks = np.stack([out_[2] for out_ in outputs])
        # AOI_ratios = np.stack([out_[3] for out_ in outputs])
        # invalid_shapes = pd.concat([out_[4] for out_ in outputs])
        #
        # labels = AOI_labels.max(axis=0)
        # masks = AOI_masks.max(axis=0)
        # ids = AOI_ids.sum(axis=0)
        #
        # locs = np.stack(np.where((AOI_labels > 0).sum(axis=0) > 1)).T
        #
        # for i, loc in enumerate(locs):
        #
        #     if i % 1000 == 0:
        #         print("correcting inter process overlaps, step %d of %d" % (i, locs.shape[0]))
        #
        #     if any(AOI_ratios[:, loc[0], loc[1]] == 1.0):
        #         masks[loc[0], loc[1]] = 2
        #     else:
        #         masks[loc[0], loc[1]] = 1
        #
        #     idx = np.argmax(AOI_ratios[:, loc[0], loc[1]])
        #     labels[loc[0], loc[1]] = AOI_labels[idx, loc[0], loc[1]]
        #     ids[loc[0], loc[1]] = AOI_ids[idx, loc[0], loc[1]]
        #
        # np.savetxt("%s/LABELS_Y%s_N%s_W%s_R%d_CRS%s.csv" %
        #            (savedir, str(year), str(maxy), str(minx), res, str(crs)), labels)
        # np.savetxt("%s/IDS_Y%s_N%s_W%s_R%d_CRS%s.csv" %
        #            (savedir, str(year), str(maxy), str(int(maxx)), res, str(crs)), ids)
        # np.savetxt("%s/MASKS_Y%s_N%s_W%s_R%d_CRS%s.csv" %
        #            (savedir, str(year), str(maxy), str(int(maxx)), res, str(crs)), masks)
        # if invalid_shapes.shape[0] != 0:
        #     invalid_shapes.to_csv(
        #         "%s/INVALID_Y%s_N%s_W%s_R%d_CRS%s.csv" %
        #         (savedir, str(year), str(maxy), str(int(maxx)), res, str(crs)), index=False)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Make raster from shapely polygons')
    # parser.add_argument('--ground_truths_file', help='filename containing ground truth parcel polygons')
    # parser.add_argument('--products_dir', help='directory containing sentinel products')
    # parser.add_argument('--savedir', help='save directory to extract ground truths in raster mode')
    # parser.add_argument('--res', default=10, help='pixel size in meters')
    # parser.add_argument('--sample_size', default=24, help='spatial resolution of dataset samples')
    # parser.add_argument('--num_processes', default=4, help='number of parallel processes')
    #
    # args = parser.parse_args()
    #
    # ground_truths_file = args.ground_truths_file
    #
    # products_dir = args.products_dir
    #
    # savedir = args.savedir
    # print("savedir: ", savedir)
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    #
    # res = int(args.res)
    #
    # sample_size = int(args.sample_size)
    #
    # num_processes = int(args.num_processes)
    #
    #
    # main()

    ground_truths_file = '/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18/gt_df_parcels_in_AOI.csv'
    products_dir = '/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/cloud_0_30'
    savedir = '/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18/LABELS4'
    res = 10
    sample_size = 100  # 64
    num_processes = 4

    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    #
    # main()
