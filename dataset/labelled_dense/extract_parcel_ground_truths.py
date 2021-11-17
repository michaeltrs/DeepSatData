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
from utils.geospatial_data_utils import GeoTransform, get_points_from_str_poly, simplify_poly_points, is_valid, str_line_eq
from utils.multiprocessing_utils import split_df
from utils.sentinel_products_utils import get_S2prod_info
import pickle
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
from utils.geospatial_data_utils import plot_poly
from copy import deepcopy


# def plot_poly(points, c=None, newfig=False):
#     if type(points) in [list, tuple]:
#         points = np.array(points)
#     if c is None:
#         c = "r"
#     if newfig:
#         plt.figure()
#     for i in range(points.shape[0] - 1):
#         plt.plot(points[i:i + 2, 0], points[i:i + 2, 1], c=c)


def extract_parcel_labels_raster(inputs):

    # rank = 0
    # inputs = inputs[0]
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
        # ii = 3600 #   4500 #100  #35 # 5200  #
        print("process %d, parcel %d of %d" % (rank, ii+1, geodata.shape[0]))
        parcel_poly = geodata['geometry'][ii]
        label = geodata['ground_truth'][ii]
        id = geodata['id'][ii]

        points = get_points_from_str_poly(parcel_poly)
        anchor = np.array(geometry.Polygon(points).centroid)  # anchor is centroid of parcel
        # anchor = points.mean(axis=0)
        N0 = anchor[1] + sample_size * res / 2.  # Nmax of image
        W0 = anchor[0] - sample_size * res / 2.  # Wmin of image

        # correct for non integer offset wrt product Nmax, Wmax (top-left) coordinates
        dN = (Np - N0) % 60
        dW = (W0 - Wp) % 60
        N0 += dN
        W0 -= dW
        # anchor[1] = N0 - sample_size * res / 2.
        # anchor[0] = W0 + sample_size * res / 2.
        anchor = np.array([W0 + sample_size * res / 2., N0 - sample_size * res / 2.])  # recalculate centroid
        # anchor = points.mean(axis=0) #- sample_size * res / 2

        # pr = points - anchor
        pr = (points - anchor + sample_size * res / 2)  # local polygon coordinates
        parcel_poly = geometry.Polygon(pr)

        ### Define criterion for removing very slender fields
        slenderness = parcel_poly.area / parcel_poly.length  # 1.0
        if slenderness < 5:
            continue

        # min, max coordinates
        pxmin, pymin = pr.min(axis=0)
        pxmax, pymax = pr.max(axis=0)

        # DONT DO VERY SMALL ONES
        if ((pxmax - pxmin) < 50) or ((pymax - pymin) < 50):
            continue

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

        # define zero placeholder matrices
        # labels = np.zeros((sample_size, sample_size), dtype=np.float32)
        # ids = labels.copy()
        ratios = np.zeros((sample_size, sample_size), dtype=np.float32)
        alpha = ratios.copy()
        # global_alpha = ratios.copy()
        global_beta = ratios.copy()
        # local_alpha = ratios.copy()
        local_beta = ratios.copy()

        # include 2 pixel threshold
        row0 = int(np.floor((pymin / (sample_size * res)) * sample_size)) - 2 # min row containing parcel
        row1 = int(np.ceil((pymax / (sample_size * res)) * sample_size)) + 2 # max row containing parcel
        col0 = int(np.floor(pxmin / (sample_size * res) * sample_size)) - 2 # min col containing parcel
        col1 = int(np.ceil(pxmax / (sample_size * res) * sample_size)) + 2 # max col containing parcel
        # row0 = int(np.floor((1 - pymax / (sample_size * res)) * sample_size))  # min row containing parcel
        # row1 = int(np.ceil((1 - pymin / (sample_size * res)) * sample_size))  # max row containing parcel
        # col0 = int(np.floor(pxmin / (sample_size * res) * sample_size))  # min col containing parcel
        # col1 = int(np.ceil(pxmax / (sample_size * res) * sample_size))  # max col containing parcel
        # row0 = int((1 - pr[:, 1].max() / dy) * AOI_labels.shape[0])
        # row1 = int((1 - pr[:, 1].min() / dy) * AOI_labels.shape[0])
        # col0 = int(pr[:, 0].min() / dx * AOI_labels.shape[1])
        # col1 = int(pr[:, 0].max() / dx * AOI_labels.shape[1]) + 1

        # H, W = sample_size, sample_size
        Height, Width = row1 - row0, col1 - col0

        # if (Height < 5) or (Width)
        # discard = False

        for i in range(Height):
            # if discard:
            #     break
            for j in range(Width):
                # if discard:
                #     break
                # i, j = 2, 2  # 45, 50
                # i, j = 5, 10
                # i, j = 9, 3
                # 48, 44
                if (row0 + i) * (col0 + j) < 0:
                    continue

                try:

                    # pix_points = [[pxmin + loc[0] * res, pymax - loc[1] * res] for loc in
                    #               [[j, i], [j + 1, i], [j + 1, i + 1], [j, i + 1], [j, i]]]
                    # pix_points = [[res * (row0 + i + 1 + loc[0]), res * (col0 + j + 1 + loc[1])] for loc in
                    #               [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]]
                    pix_points = [[res * (col0 + j + 1 + loc[1]), res * (row0 + i + 1 + loc[0])] for loc in
                                  [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]]

                    pix_poly = geometry.Polygon(pix_points)

                    value = parcel_poly.intersection(pix_poly).area / res ** 2
                    # print(value)
                    if (0 < value) and (value < 1):  # parcel cuts through pixel
                        # discard = True
                        # print(i, j)
                        # continue
                        global_points = np.array(parcel_poly.boundary.intersection(pix_poly.boundary))
                        # np.array(parcel_poly.exterior.coords)
                        if global_points.shape[0] > 2:  # !!!
                            # print(i, j)
                            global_points = global_points[:2]
                        global_params = str_line_eq(global_points)
                        alpha[sample_size - (row0 + i + 1), col0 + j + 1] = global_params[0]
                        # global_alpha[row0 + i + 1, col0 + j + 1] = global_params[0]
                        global_beta[sample_size - (row0 + i + 1), col0 + j + 1] = global_params[1] / (sample_size * res)

                        # local_points = (global_points - np.array([pxmin + j * res, pymax - i * res])) / res
                        local_points = (global_points - np.array([res * (col0 + j + 0.5),  res * (row0 + i + 0.5)])) / res

                        local_params = str_line_eq(local_points)
                        # local_alpha[row0 + i + 1, col0 + j + 1] = local_params[0]
                        local_beta[sample_size - (row0 + i + 1), col0 + j + 1] = local_params[1]

                        # break

                    if value == 0:  # no intersection
                        continue

                    # labels[row0 + i + 1, col0 + j + 1] = label
                    ratios[sample_size - (row0 + i + 1), col0 + j + 1] = value
                    # ratios[col0 + i + 1, row0 + j + 1] = value

                    # ids[row0 + i + 1, col0 + j + 1] = id

                except:
                    # print(i, j)
                    # discard = True
                    continue

        # if discard:
        #     continue

        idxN = int(np.round((N_ - N0) / R_ - 1., 0)) #- 1
        idxW = int(np.round((W0 - W_) / R_ - 1., 0)) #- 1


        # add AOI raster ground truths
        labels2d = raster['LABELS'][idxN:idxN+sample_size, idxW:idxW+sample_size]
        ids2d = raster['IDS'][idxN:idxN+sample_size, idxW:idxW+sample_size]
        masks2d = raster['MASKS'][idxN:idxN+sample_size, idxW:idxW+sample_size]
        distances2d = raster['DISTANCES'][idxN:idxN+sample_size, idxW:idxW+sample_size]
        ratios2d = raster['RATIOS'][idxN:idxN+sample_size, idxW:idxW+sample_size]

        # add simpilied polygons
        simplified = simplify_poly_points(pr, Npoly)

        # replace global, local alpha with alpha
        # sample = {'N': N0, 'W': W0, 'boundary': pr / res, 'label': label, 'id': id, 'ratios': ratios,
        #           'alpha': alpha, 'global_beta': global_beta, 'local_beta': local_beta}
        sample = {'N': N0, 'W': W0,
                  'poly_var': pr / res, 'poly_fixed': simplified / res,
                  'label': label, 'id': id,
                  'labels2d': labels2d, 'ids2d': ids2d, 'masks2d': masks2d, 'distances2d': distances2d,
                  'ratios': ratios, 'alpha': alpha, 'global_beta': global_beta, 'local_beta': local_beta}

        impath = os.path.join(year_savedir, 'N%d_E%d_ground_truths.pickle' % (N0, W0))
        with open(impath, 'wb') as handle:
            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

        values = geodata.iloc[ii].to_list()
        for v in [N0, W0, impath]:
            values.append(v)
        saved_data_info.append(values)

    return saved_data_info


# plt.figure()
# # plt.imshow(sample['local_beta'])
# # plt.imshow(local_beta)
# plt.imshow(alpha)
# # plt.imshow(global_beta)
# plt.title('local_beta')
#
# pix = np.array(pix_points) / res
# pix[:, 1] = 100 - pix[:, 1]
# plot_poly(pix, c='b', newfig=False)
# s = simplified / res
# s[:, 1] = 100 - s[:, 1]
# plot_poly(s, c='k', newfig=False)
#
# plt.scatter(global_points[:, 0]/res, 100-global_points[:, 1]/res, c='r')

# #
# #
# #
# # # plot_poly(sample['poly_var'], newfig=True)
# # pol = sample['poly_fixed']
# # pol[:, 1] = 100 - pol[:, 1]
# plot_poly(sample['poly_fixed'], c='r', newfig=False)
#
#       plt.scatter(global_points[:, 0] / res, global_points[:, 1] / res, c='b')
# plt.figure()
# plt.imshow(sample['labels2d'])
# plt.title('labels2d')

# plt.figure()
# plt.imshow(sample['ids2d'])
# plt.title('ids2d')
# #
# # plt.figure()
# # plt.imshow(sample['masks2d'])
# # plt.title('masks2d')
# #
# # plt.figure()
# # plt.imshow(sample['distances2d'])
# # plt.title('distances2d')
# #
# plt.figure()
# plt.imshow(sample['ratios'])
# plt.title('ratios')
#
# plt.figure()
# plt.imshow(ratios2d)
# plt.title('ratios')
#
# plt.figure()
# plt.imshow(ratios2d)
# plt.title('ratios2d')
#
# plt.figure()
# plt.imshow(sample['alpha'])
# plt.title('alpha')
# plot_poly(sample['poly_var'][:, ::-1], newfig=False)
#
# poly = pr.copy()
# p = simplify_poly_points(poly, numpoints=200)#.shape

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
    #
    # l = labels2d.copy()
    # l[l==2]=100
    # plt.figure()
    # plt.imshow(l)
    #
    #
    # plot_poly(pr / res, c='b', newfig=True)
    # plot_poly(simplified, newfig=False)
    # plot_poly(simplified2, c='k', newfig=False)
    #
    # plot_poly(p, c='b', newfig=True)

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

# plot_poly(np.array(parcel_poly.exterior.coords.xy).T, newfig=False)
# plt.scatter(pix_center[0], pix_center[1])
#
# p2 = closest_point_to_poly(np.array(parcel_poly.exterior.coords.xy).T, pix_center)
# plt.scatter(p2[0], p2[1])
# plt.gca().set_aspect('equal', adjustable='box')
#
# d = AOI_distances[6767:6800, 5081:5115]
# plt.imshow(d/d.max())
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
    global N_
    global W_
    global R_
    global CRS_
    global raster

    # ground truth data
    gt_df = pd.read_csv(ground_truths_file)
    # gt_df = gt_df.iloc[:5000]  # REMOVE!!!
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

    # AOI rasterized ground truths
    raster_files = [fname for fname in os.listdir(raster_labels_dir) if fname.endswith('csv')]
    raster = {}
    meta = []
    for raster_file in raster_files:
        # raster_file = raster_files[0]
        ftype_ = raster_file.split("_")[0]
        year_ = raster_file.split("_")[1][1:]
        N_ = raster_file.split("_")[2][1:]
        W_ = raster_file.split("_")[3][1:]
        R_ = raster_file.split("_")[4][1:]
        CRS_ = raster_file.split("_")[5][3:].split('.')[0]
        raster[ftype_] = np.loadtxt(os.path.join(raster_labels_dir, raster_file))
        meta.append([year_, N_, W_, R_, CRS_])
    meta = np.array(meta)
    assert all([(meta[i] == meta[0]).all() for i in range(len(meta))]), \
        'Not all AOI raster ground truth files correspond to the same location or time'
    N_ = int(N_)
    W_ = int(W_)
    R_ = int(R_)
    CRS_ = int(CRS_)

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

    parser = argparse.ArgumentParser(description='Make raster from shapely polygons')
    parser.add_argument('--ground_truths_file', help='filename containing ground truth parcel polygons')
    parser.add_argument('--raster_labels_dir', help='directory containing extracted raster ground truths')
    parser.add_argument('--products_dir', help='directory containing sentinel products')
    parser.add_argument('--savedir', help='save directory to extract ground truths in raster mode')
    parser.add_argument('--res', default=10, help='pixel size in meters')
    parser.add_argument('--sample_size', default=24, help='spatial resolution of dataset samples')
    parser.add_argument('--Npoly', default=50, help='number of vertices for polygons')
    parser.add_argument('--num_processes', default=4, help='number of parallel processes')

    args = parser.parse_args()

    ground_truths_file = args.ground_truths_file

    raster_labels_dir = args.raster_labels_dir

    products_dir = args.products_dir

    savedir = args.savedir
    print("savedir: ", savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    res = int(args.res)

    sample_size = int(args.sample_size)

    Npoly = int(args.Npoly)

    num_processes = int(args.num_processes)

    main()


    # ground_truths_file = '/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18_3/gt_df_parcels_in_AOI.csv'
    # raster_labels_dir = '/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18_3/LABELS'
    # products_dir = '/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/cloud_0_30'
    # savedir = '/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18_3/LABELS'
    # res = 10
    # sample_size = 100  # 64
    # num_processes = 4
    # Npoly = 50

    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)

    # main()
