import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def get_nbins(x):
    N = x.shape[0]
    if N < 1e2:
        return 10
    if N < 1e3:
        return 25
    # if N < 1e4:
    else:
        return 100


def main():
    # ground truth data
    gt_df = pd.read_csv(ground_truths_file)
    # gt_df['id'] = range(1, gt_df.shape[0]+1)
    # gt_df['crs'] = 2154
    if 'id' not in gt_df:
        gt_df['id'] = range(1, gt_df.shape[0]+1)
    assert (gt_df['crs'] == gt_df['crs'].iloc[0]).all(), \
        "Polygons corresponding to multiple CRS were found in %s" % ground_truths_file
    crs = gt_df['crs'].iloc[0]
    yearly_grouped_gt = gt_df.groupby('year')
    years = list(yearly_grouped_gt.groups.keys())
    print("found ground truth data for years %s" % ", ".join([str(i) for i in years]))
    # if 0 in gt_df['ground_truth'].drop_duplicates():
    #     gt_df['ground_truth'] += 1

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
    print(prod_poly)
    def f(x):
        try:
            x = get_points_from_str_poly(x)
            W = x[:, 0].min()
            E = x[:, 0].max()
            S = x[:, 1].min()
            N = x[:, 1].max()
            num_vertices = x.shape[0]
            x = geometry.Polygon(x)
            inratio = prod_poly.intersection(x).area / x.area
            return np.array([N, S, W, E, inratio, num_vertices])
        except:
            return np.array([0, 0, 0, 0, 0, 0])

    gt_df[['N', 'S', 'W', 'E', 'inratio', 'num_vertices']] = np.stack(gt_df['geometry'].apply(f).values)
    gt_df = gt_df[gt_df['inratio'] == 1.0]
    print("found %d polygons inside sentinel tile" % gt_df.shape[0])

    gt_df['Dy'] = np.abs(gt_df['N'] - gt_df['S'])
    gt_df['Dx'] = np.abs(gt_df['E'] - gt_df['W'])
    gt_df['D'] = gt_df[['Dx', 'Dy']].max(axis=1)
    # gt_df['D'].max()
    # gt_df[gt_df['D'] < 480].shape[0] / gt_df.shape[0]
    # gt_df[gt_df['D'] > 700].shape[0] / gt_df.shape[0]
    gt_df.to_csv(os.path.join(save_dir, 'gt_df_parcels_in_AOI.csv'), index=False)

    # if cutoff is None:
    # x = np.random.normal(mu, sigma, size=100)
    print('maxD   | %obj >maxD')
    print('-------------------')
    for maxd in [240, 320, 480, 640, 1000, 1280, 1600]:
        r = gt_df[gt_df['D'] < maxd].shape[0] / gt_df.shape[0]
        print('%s|%s' % (str(maxd).ljust(7), ('%.4f' % r).rjust(9)))
    plt.ioff()
    fig, ax = plt.subplots(figsize=(8, 4))
    n_bins = get_nbins(gt_df['D'])
    # plot the cumulative histogram
    n, bins, patches = ax.hist(gt_df['D'].values, n_bins, density=True, histtype='step',
                               cumulative=True, label='cummulative sum')
    ax.hist(gt_df['D'].values, bins=bins, density=True, histtype='step', cumulative=-1,
            label='reversed cummulative sum')
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('Object largest x-y dimension')
    ax.set_ylabel('Likelihood of occurrence')
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'parcel_dimensions_cumsum.png'))

    plt.figure()
    plt.hist(gt_df['num_vertices'], 100, density=True)
    plt.grid()
    plt.xlabel('Number of AF Vertices')
    plt.ylabel('density')
    plt.savefig(os.path.join(save_dir, 'number_of_vertices_hist.png'))

    # else:
    #     gt_df = gt_df[gt_df['D'] < cutoff]
    #     print('Number of samples is %d for max object dimension <%dm' % (gt_df.shape[0], cutoff))
    #     gt_df.to_csv(os.path.join(save_dir, 'gt_df_maxd_lt_%d.csv' % cutoff), index=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Make raster from shapely polygons')
    parser.add_argument('--ground_truths_file', help='filename containing ground truth parcel polygons')
    parser.add_argument('--products_dir', help='directory containing sentinel products')
    parser.add_argument('--save_dir', help='save directory to extract ground truths in raster mode')
    # parser.add_argument('--cutoff', default=None, help='max allowed parcel size. If None to script will save a cumsum '
    #                                                    'histogram to help decide the max alloed size')
    # parser.add_argument('--sample_size', default=24, help='spatial resolution of dataset samples')
    # parser.add_argument('--num_processes', default=4, help='number of parallel processes')

    args = parser.parse_args()

    ground_truths_file = args.ground_truths_file

    products_dir = args.products_dir

    save_dir = args.save_dir
    print("save_dir: ", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # cutoff = int(args.cutoff)

    main()
