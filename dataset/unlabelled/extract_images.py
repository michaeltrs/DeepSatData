"""
Given a directory of Sentinel tiles extract crops of images
"""
import argparse
import pandas as pd
import rasterio
import numpy as np
import os
from glob import glob
import pickle
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.append(dir(dir(path[0])))
    __package__ = "examples"
from utils.geospatial_data_utils import GeoTransform
from utils.multiprocessing_utils import run_pool
from utils.sentinel_products_utils import get_S2prod_info


mult = {'B01': 1/6., 'B02': 1., 'B03': 1., 'B04': 1., 'B05': 1./2., 'B06': 1./2., 'B07': 1./2., 'B08': 1., 'B8A': 1./2,
        'B09': 1./6., 'B10': 1./6., 'B11': 1./2., 'B12': 1./2.}
# jp2s = ["%s.jp2" % i for i in mult.keys()]


def extract_images(imdirs):

    jp2s = ["%s.jp2" % i for i in bands]
    # print('jp2s: ', jp2s)

    refband = None
    for band in bands:
        if mult[band] == 1.0:
            refband = band
            break
    assert refband is not None, "in curerent implementation at least one 10m band should be included"

    saved_files_info = []

    for ii, imdir in enumerate(imdirs):

        print("processing product %d of %d in current process" % (ii+1, len(imdirs)))

        imname = "%s_%s" % (imdir.split("/")[-2].split("_")[-3], imdir.split("/")[-4].split("_")[-5])
        date = imdir.split("/")[-4].split(".")[0].split("_")[2][:8]

        # read product
        data = {}
        for jp2 in jp2s:
            with rasterio.open("%s/%s_%s" % (imdir, imname, jp2)) as f:
                data[jp2[:-4]] = f.read(1)

        if anchor is not None:
            Nanchor, Wanchor, CRSanchor = anchor

            geotransform_prod2anchor = GeoTransform(CRSanchor, str(f.crs).split(':')[1], loc2loc=True)
            Wp, Np = geotransform_prod2anchor(np.array(f.transform)[2], np.array(f.transform)[5])

            dN = divmod((Np - Nanchor) / (sample_size * res), 1)[1] * sample_size * res
            dW = divmod((Wanchor - Wp) / (sample_size * res), 1)[1] * sample_size * res

        else:
            Wp, Np = np.array(f.transform)[2], np.array(f.transform)[5]
            dN = dW = 0

        num_rows = (data[refband].shape[0] * 10 - dN) / (sample_size * res)
        num_cols = (data[refband].shape[0] * 10 - dW) / (sample_size * res)

        prod_savedir = os.path.join(savedir, imdir.split("/")[-4].split(".")[0])
        if not os.path.exists(prod_savedir):
            os.makedirs(prod_savedir)

        for i in range(int(num_rows)):

            for j in range(int(num_cols)):

                Nij = Np - dN - i * res * sample_size  # N for extracted label window
                Wij = Wp + dW + j * res * sample_size  # W for extracted label window

                ip = (Np - Nij) / (res * sample_size)  # product row
                jp = (Wij - Wp) / (res * sample_size)  # product column

                sample = {}
                for jp2 in jp2s:
                    xpmin = int(np.round(mult[jp2[:-4]] * ip * sample_size))
                    ypmin = int(np.round(mult[jp2[:-4]] * jp * sample_size))
                    sample[jp2[:-4]] = data[jp2[:-4]][xpmin: xpmin + int(mult[jp2[:-4]] * sample_size),
                                       ypmin: ypmin + int(mult[jp2[:-4]] * sample_size)]

                if sample[jp2[:-4]].sum() == 0:
                    saved_files_info.append(
                        [None, Nij, Wij, Np, Wp, i, j, ip, jp, sample_size, sample_size, date, imdir, "no image"])
                    continue

                sample_save_path = "%s/N%d_W%d_D%s.pickle" % (prod_savedir, int(Nij), int(Wij), date)
                with open(sample_save_path, 'wb') as handle:
                    pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

                saved_files_info.append(
                    [sample_save_path, Nij, Wij, Np, Wp, i, j, ip, jp, sample_size, sample_size, date, imdir, "ok"])


    df = pd.DataFrame(data=saved_files_info,
                      columns=['sample_path', 'Nij', 'Wij', 'Np', 'Wp', 'il', 'jl', 'ip', 'jp',
                               'height', 'width', 'Date', 'S2_prod_imdir', "comment"])
    print('returned')
    return df


def main():

    # sentinel products
    imdirs = glob("%s/*.SAFE/GRANULE/**/IMG_DATA" % products_dir)
    prod_df = get_S2prod_info(imdirs)
    prod_df['Year'] = prod_df['Time'].apply(lambda s: s[:4])
    years = prod_df['Year'].drop_duplicates().tolist()

    out = []
    for year in years:

        # sentinel products
        products = prod_df[prod_df['Year'] == year]
        imdirs = products['path'].tolist()

        df_year = run_pool(imdirs, extract_images, num_processes)

        out.append(pd.concat(df_year))


    df = pd.concat(out).reset_index(drop=True)
    df.to_csv(os.path.join(savedir, "extracted_windows_data_info.csv"), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--products_dir', help='directory containing sentinel products')
    parser.add_argument('--bands', default=None, help='which satellite image bands to use')
    parser.add_argument('--savedir', help='save directory to extract ground truths in raster mode')
    parser.add_argument('--res', default=10, help='pixel size in meters')
    parser.add_argument('--anchor', default=None, help='anchor point for grid (N, W, crs)')
    parser.add_argument('--sample_size', default=24, help='spatial resolution of dataset samples')
    parser.add_argument('--num_processes', default=4, help='number of parallel processes')
    # ---------------------------------------------------------------------------------------------

    args = parser.parse_args()
    products_dir = args.products_dir
    bands = args.bands
    if bands == 'None':
        bands = list(mult.keys())
    else:
        bands = bands.split(',')
    savedir = args.savedir
    res = int(args.res)
    sample_size = int(args.sample_size)
    num_processes = int(args.num_processes)
    anchor = args.anchor
    if anchor == 'None':
        anchor = None
    else:
        anchor = [int(i) for i in anchor.split(",")]

    main()
