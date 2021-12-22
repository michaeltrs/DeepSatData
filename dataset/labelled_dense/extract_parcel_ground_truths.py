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


def extract_parcel_labels_raster(inputs):

    # inputs = inputs[0]
    rank, geodata, W, N, Wp, Np, year, crs = inputs

    # arrays to save
    year_savedir = os.path.join(savedir, 'Y%s_N%s_W%s_R%d_CRS%s' % (year, N, W, res, crs))
    if not os.path.exists(year_savedir):
        os.makedirs(year_savedir)

    saved_data_info = []
    for ii in range(geodata.shape[0]):
        print("process %d, parcel %d of %d" % (rank, ii+1, geodata.shape[0]))
        parcel_poly = geodata['geometry'][ii]
        label = geodata['ground_truth'][ii]
        id = geodata['id'][ii]

        points = get_points_from_str_poly(parcel_poly)
        anchor = np.array(geometry.Polygon(points).centroid)  # anchor is centroid of parcel
        # anchor = points.mean(axis=0)
        N0 = anchor[1] + sample_size * 10. / 2.  # Nmax of image
        W0 = anchor[0] - sample_size * 10. / 2.  # Wmin of image

        # correct for non integer offset wrt product Nmax, Wmax (top-left) coordinates
        dN = (Np - N0) % 60
        dW = (W0 - Wp) % 60
        N0 += dN
        W0 -= dW
        anchor = np.array([W0 + sample_size * 10. / 2., N0 - sample_size * 10. / 2.])  # recalculate centroid

        pr = (points - anchor + sample_size * 10. / 2)  # local polygon coordinates
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
        ratios = np.zeros((label_mult * sample_size, label_mult * sample_size), dtype=np.float32)
        alpha = ratios.copy()
        global_beta = ratios.copy()
        local_beta = ratios.copy()

        # include 2 pixel threshold (this wont matter as external pixels will not update their values)
        row0 = int(np.floor((pymin / (sample_size * 10)) * label_mult * sample_size)) - 2  # min row containing parcel
        row1 = int(np.ceil((pymax / (sample_size * 10)) * label_mult * sample_size)) + 2  # max row containing parcel
        col0 = int(np.floor(pxmin / (sample_size * 10) * label_mult * sample_size)) - 2  # min col containing parcel
        col1 = int(np.ceil(pxmax / (sample_size * 10) * label_mult * sample_size)) + 2  # max col containing parcel

        Height, Width = row1 - row0, col1 - col0

        for i in range(Height):

            for j in range(Width):

                if (row0 + i) * (col0 + j) < 0:
                    continue

                try:

                    pix_points = [[res * (col0 + j + loc[1]), res * (row0 + i + loc[0])] for loc in
                                  [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]]

                    pix_poly = geometry.Polygon(pix_points)

                    value = parcel_poly.intersection(pix_poly).area / res ** 2
                    if (0 < value) and (value < 1):  # parcel cuts through pixel
                        global_points = np.array(parcel_poly.boundary.intersection(pix_poly.boundary))
                        if global_points.shape[0] > 2:  # !!!
                            global_points = global_points[:2]
                        global_params = str_line_eq(global_points)
                        alpha[label_mult * sample_size - (row0 + i + 1), col0 + j + 1] = global_params[0]
                        global_beta[label_mult * sample_size - (row0 + i + 1), col0 + j + 1] = global_params[1] / (sample_size * res)
                        local_points = (global_points - np.array([res * (col0 + j + 0.5),  res * (row0 + i + 0.5)])) / res
                        local_params = str_line_eq(local_points)
                        local_beta[label_mult * sample_size - (row0 + i + 1), col0 + j + 1] = local_params[1]


                    if value == 0:  # no intersection
                        continue

                    ratios[label_mult * sample_size - (row0 + i + 0), col0 + j + 0] = value

                except:
                    continue

        idxN = int(np.round((N_ - N0) / res - 1., 0))
        idxW = int(np.round((W0 - W_) / res - 1., 0))

        # add AOI raster ground truths
        labels2d = raster['LABELS'][idxN: idxN + label_mult * sample_size, idxW: idxW + label_mult * sample_size]
        ids2d = raster['IDS'][idxN: idxN + label_mult * sample_size, idxW: idxW + label_mult * sample_size]
        masks2d = raster['MASKS'][idxN: idxN + label_mult * sample_size, idxW: idxW + label_mult * sample_size]
        distances2d = raster['DISTANCES'][idxN: idxN + label_mult * sample_size, idxW: idxW + label_mult * sample_size]
        ratios2d = raster['RATIOS'][idxN: idxN + label_mult * sample_size, idxW: idxW + label_mult * sample_size]

        # add simpilied polygons
        simplified = simplify_poly_points(pr, Npoly)

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


def main():
    global N_
    global W_
    global R_
    global CRS_
    global raster
    global label_mult

    label_mult = int(10. / res)

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
        'Not all AOI raster ground truth files correspond to the same location, time, resolution or crs'
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

    N = int(np.ceil(gt_df['N'].max()))
    W = int(np.floor(gt_df['W'].min()))

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

    # res = int(args.res)
    res = float(args.res)
    assert np.ceil(10. / res) == 10. / res, "Label pixel size should divide min satellite pixel size (10m), but %.1f was selected" % res

    sample_size = int(args.sample_size)

    Npoly = int(args.Npoly)

    num_processes = int(args.num_processes)

    main()
