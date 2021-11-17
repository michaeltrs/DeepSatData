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
from utils.geospatial_data_utils import GeoTransform, get_points_from_str_poly, closest_point_to_poly
from utils.multiprocessing_utils import split_df
from utils.sentinel_products_utils import get_S2prod_info


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


def extract_labels_raster(inputs):
    # inputs = inputs[0]
    rank, geodata, anchor, dx, dy = inputs

    # arrays to save
    AOI_labels = np.zeros((int(np.round(dy / res)), int(np.round(dx / res))), dtype=np.float32) # + max_label + 1
    AOI_ids = np.zeros((int(np.round(dy / res)), int(np.round(dx / res))), dtype=np.float32)
    AOI_masks = AOI_ids.copy()
    # additional/helper arrays
    AOI_ratios = AOI_ids.copy()
    AOI_distances = AOI_ids.copy()
    # AOI_alphas = AOI_ids.copy()

    invalid_shapes = []
    for ii in range(geodata.shape[0]):
        # ii = 0
        print("process %d, parcel %d of %d" % (rank, ii+1, geodata.shape[0]))
        parcel_poly = geodata['geometry'][ii]
        label = geodata['ground_truth'][ii]
        id = geodata['id'][ii]

        points = get_points_from_str_poly(parcel_poly)
        pr = (points - anchor)
        parcel_poly = geometry.Polygon(pr)

        pxmin, pymin = pr.min(axis=0)
        pxmax, pymax = pr.max(axis=0)

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
                    invalid_shapes.append(geodata.iloc[ii])
                    continue
            except:
                continue

        row0 = int((1 - pr[:, 1].max() / dy) * AOI_labels.shape[0])
        row1 = int((1 - pr[:, 1].min() / dy) * AOI_labels.shape[0])
        col0 = int(pr[:, 0].min() / dx * AOI_labels.shape[1])
        col1 = int(pr[:, 0].max() / dx * AOI_labels.shape[1]) + 1

        H, W = row1 - row0, col1 - col0

        for i in range(H):

            for j in range(W):
                # i = j = 15
                try:

                    pix_points = [[pxmin + loc[0] * res, pymax - loc[1] * res] for loc in
                                  [[j, i], [j + 1, i], [j + 1, i + 1], [j, i + 1], [j, i]]]

                    pix_poly = geometry.Polygon(pix_points)

                    value = parcel_poly.intersection(pix_poly).area / res ** 2

                    if value == 0:  # no intersection
                        continue

                    elif AOI_ratios[row0 + i, col0 + j] == 1.0:  # interior of at least another poly

                        if AOI_labels[row0 + i, col0 + j] != label:  # mask only if label conflict
                            AOI_masks[row0 + i, col0 + j] = 2
                        continue

                    elif AOI_ratios[row0 + i, col0 + j] > 0:  # at least partly assigned to another poly
                        if AOI_labels[row0 + i, col0 + j] != label:  # mask only if label conflict
                           AOI_masks[row0 + i, col0 + j] = 1

                    if value > AOI_ratios[row0 + i, col0 + j]:  # this poly covers a larger area, assign here
                        AOI_labels[row0 + i, col0 + j] = label
                        AOI_ratios[row0 + i, col0 + j] = value
                        AOI_ids[row0 + i, col0 + j] = id
                        pix_center = np.array(pix_points)[:-1].mean(axis=0)
                        AOI_distances[row0 + i, col0 + j] = closest_point_to_poly(
                            np.array(parcel_poly.exterior.coords.xy).T, pix_center, return_dist=True)

                except:
                    continue

    return AOI_labels, AOI_ids, AOI_masks, AOI_ratios, AOI_distances, pd.DataFrame(invalid_shapes)


def main():
    # ground truth data
    gt_df = pd.read_csv(ground_truths_file)

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
    geotr = GeoTransform(intr=prod_df['crs'].iloc[0].split(':')[1], outtr=gt_df['crs'].iloc[0], loc2loc=True)
    prod_WN = prod_df[['West', 'North']].iloc[0].tolist()
    prod_WN = geotr(prod_WN[0], prod_WN[1])  # in ground truth data coordinate system
    d = (10 * prod_df[['height', 'width']].iloc[0].values).tolist()

    # find all ground truth data that fall inside sentinel product
    prod_poly = geometry.Polygon([[prod_WN[0] + loc[0] * d[0], prod_WN[1] - loc[1] * d[1]] for loc in
                                  [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]])
    # print(prod_poly)
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

    maxy = int(np.ceil(gt_df['N'].max()))   # N
    miny = int(np.floor(gt_df['S'].min()))  # S!
    maxx = int(np.ceil(gt_df['E'].max()))   # E!
    minx = int(np.floor(gt_df['W'].min()))  # W

    # increase AOI dimensions to match integer multiple of sample size
    if np.ceil((maxy - miny) / (sample_size * res)) != (maxy - miny) / (sample_size * res):
        dy = (np.ceil((maxy - miny) / (sample_size * res)) - (maxy - miny) / (sample_size * res)) * (sample_size * res)
        miny = miny - dy
    if np.ceil((maxx - minx) / (sample_size * res)) != (maxx - minx) / (sample_size * res):
        dx = (np.ceil((maxx - minx) / (sample_size * res)) - (maxx - minx) / (sample_size * res)) * (sample_size * res)
        maxx = maxx + dx
    dx = maxx - minx
    dy = maxy - miny
    anchor = minx, miny  # WS

    pool = Pool(num_processes)

    for year in years:
        # year = years[0]

        geodata = gt_df[gt_df['year'] == year].reset_index(drop=True)

        inputs = [[i, df_, anchor, dx, dy] for i, df_ in enumerate(split_df(geodata, num_processes))]

        outputs = pool.map(extract_labels_raster, inputs)
        AOI_labels = np.stack([out_[0] for out_ in outputs])
        AOI_ids = np.stack([out_[1] for out_ in outputs])
        AOI_masks = np.stack([out_[2] for out_ in outputs])
        AOI_ratios = np.stack([out_[3] for out_ in outputs])
        AOI_distances = np.stack([out_[4] for out_ in outputs])
        invalid_shapes = pd.concat([out_[5] for out_ in outputs])

        labels = AOI_labels.max(axis=0)
        masks = AOI_masks.max(axis=0)
        ids = AOI_ids.sum(axis=0)
        ratios = AOI_ratios.max(axis=0)
        distances = AOI_distances.max(axis=0)

        locs = np.stack(np.where((AOI_labels > 0).sum(axis=0) > 1)).T

        for i, loc in enumerate(locs):

            if i % 1000 == 0:
                print("correcting inter process overlaps, step %d of %d" % (i, locs.shape[0]))

            if any(AOI_ratios[:, loc[0], loc[1]] == 1.0):
                masks[loc[0], loc[1]] = 2
            else:
                masks[loc[0], loc[1]] = 1

            idx = np.argmax(AOI_ratios[:, loc[0], loc[1]])
            labels[loc[0], loc[1]] = AOI_labels[idx, loc[0], loc[1]]
            ids[loc[0], loc[1]] = AOI_ids[idx, loc[0], loc[1]]
            ratios[loc[0], loc[1]] = AOI_ratios[idx, loc[0], loc[1]]
            distances[loc[0], loc[1]] = AOI_distances[idx, loc[0], loc[1]]

        np.savetxt("%s/LABELS_Y%s_N%s_W%s_R%d_CRS%s.csv" %
                   (savedir, str(year), str(maxy), str(minx), res, str(crs)), labels)
        np.savetxt("%s/IDS_Y%s_N%s_W%s_R%d_CRS%s.csv" %
                   (savedir, str(year), str(maxy), str(int(minx)), res, str(crs)), ids)
        np.savetxt("%s/MASKS_Y%s_N%s_W%s_R%d_CRS%s.csv" %
                   (savedir, str(year), str(maxy), str(int(minx)), res, str(crs)), masks)
        np.savetxt("%s/RATIOS_Y%s_N%s_W%s_R%d_CRS%s.csv" %
                   (savedir, str(year), str(maxy), str(int(minx)), res, str(crs)), ratios)
        np.savetxt("%s/DISTANCES_Y%s_N%s_W%s_R%d_CRS%s.csv" %
                   (savedir, str(year), str(maxy), str(int(minx)), res, str(crs)), distances)
        if invalid_shapes.shape[0] != 0:
            invalid_shapes.to_csv(
                "%s/INVALID_Y%s_N%s_W%s_R%d_CRS%s.csv" %
                (savedir, str(year), str(maxy), str(int(maxx)), res, str(crs)), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Make raster from shapely polygons')
    parser.add_argument('--ground_truths_file', help='filename containing ground truth parcel polygons')
    parser.add_argument('--products_dir', help='directory containing sentinel products')
    parser.add_argument('--savedir', help='save directory to extract ground truths in raster mode')
    parser.add_argument('--res', default=10, help='pixel size in meters')
    parser.add_argument('--sample_size', default=24, help='spatial resolution of dataset samples')
    parser.add_argument('--num_processes', default=4, help='number of parallel processes')

    args = parser.parse_args()

    ground_truths_file = args.ground_truths_file

    products_dir = args.products_dir

    savedir = args.savedir
    print("savedir: ", savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    res = int(args.res)

    sample_size = int(args.sample_size)

    num_processes = int(args.num_processes)

    main()
