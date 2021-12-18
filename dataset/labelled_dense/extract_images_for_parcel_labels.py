"""
Given a set of S2 tiles and a labelled_dense lable map, extract crops of images matching the location of labels
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
    path.insert(0, dir(dir(path[0])))
    __package__ = "examples"
from utils.data_utils import find_number
from utils.geospatial_data_utils import GeoTransform
from utils.multiprocessing_utils import run_pool
from utils.sentinel_products_utils import get_S2prod_info


mult = {'B01': 1/6., 'B02': 1., 'B03': 1., 'B04': 1., 'B05': 1./2., 'B06': 1./2., 'B07': 1./2., 'B08': 1., 'B8A': 1./2,
        'B09': 1./6., 'B10': 1./6., 'B11': 1./2., 'B12': 1./2.}


def extract_images(imdirs):

    jp2s = ["%s.jp2" % i for i in bands]

    saved_files_info = []

    for ii, imdir in enumerate(imdirs):

        print("unfolding product %d of %d" % (ii, len(imdirs)))

        imname = "%s_%s" % (imdir.split("/")[-2].split("_")[-3], imdir.split("/")[-4].split("_")[-5])

        # read product
        data = {}
        for jp2 in jp2s:
            with rasterio.open("%s/%s_%s" % (imdir, imname, jp2)) as f:
                data[jp2[:-4]] = f.read(1)

        geotransform_label2prod = GeoTransform(CRSl, str(f.crs).split(':')[1], loc2loc=CRSl != '4326')
        Wp, Np = np.array(f.transform)[2], np.array(f.transform)[5]

        prod_savedir = os.path.join(savedir, imdir.split("/")[-4].split(".")[0])
        if not os.path.exists(prod_savedir):
            os.makedirs(prod_savedir)

        for i in range(saved_gt_info.shape[0]):

            Nl = saved_gt_info.iloc[i]['Ntl']
            Wl = saved_gt_info.iloc[i]['Wtl']
            Wlp, Nlp = geotransform_label2prod(Wl, Nl)

            ip = int(np.round((Np - Nlp) / 10.))
            jp = int(np.round((Wlp - Wp) / 10.))

            date = imdir.split("/")[-4].split(".")[0].split("_")[2][:8]

            sample = {}
            for jp2 in jp2s:
                xpmin = int(np.round(mult[jp2[:-4]] * ip))
                ypmin = int(np.round(mult[jp2[:-4]] * jp))
                sample[jp2[:-4]] = data[jp2[:-4]][xpmin: xpmin + int(mult[jp2[:-4]] * sample_size),
                                                  ypmin: ypmin + int(mult[jp2[:-4]] * sample_size)]

            # this parcel falls in black region for this product
            if sample[jp2[:-4]].sum() == 0:
                saved_files_info.append(
                    ["", Nlp, Wlp, Nl, Wl, Np, Wp, ip, jp, sample_size, sample_size, date, imdir, "no image"])
                continue

            sample_save_path = "%s/N%d_W%d_D%s_CRS%s.pickle" % (prod_savedir, int(Nl), int(Wl), date, CRSl)
            with open(sample_save_path, 'wb') as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

            saved_files_info.append(
                [sample_save_path, Nlp, Wlp, Nl, Wl, Np, Wp, ip, jp, sample_size, sample_size, date, imdir, "ok"])

    df = pd.DataFrame(data=saved_files_info,
                      columns=['sample_path', 'Nlp', 'Wlp', 'Nl', 'Wl', 'Np', 'Wp', 'ip', 'jp',
                               'height', 'width', 'Date', 'S2_prod_imdir', "comment"])
    return df


def main():
    # ground truths
    gtdirs = [f for f in os.listdir(ground_truths_dir) if os.path.isdir(os.path.join(ground_truths_dir, f))]

    global CRSl
    global saved_gt_info

    # sentinel products
    imdirs = glob("%s/*.SAFE/GRANULE/**/IMG_DATA" % products_dir)
    prod_df = get_S2prod_info(imdirs)
    prod_df['Year'] = prod_df['Time'].apply(lambda s: s[:4])

    out = []
    for gtdir in gtdirs:

        # ground truths
        saved_gt_info = pd.read_csv(os.path.join(ground_truths_dir, gtdir, 'saved_data_info.csv'))

        year = find_number(gtdir, "Y")
        CRSl = find_number(gtdir, "CRS")

        # sentinel products
        products = prod_df[prod_df['Year'] == year]
        imdirs = products['path'].tolist()

        df_year = run_pool(imdirs, extract_images, num_processes)

        out.append(pd.concat(df_year))

    df = pd.concat(out).reset_index(drop=True)
    df['crs'] = CRSl
    df.to_csv(os.path.join(savedir, "extracted_windows_data_info.csv"), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--ground_truths_dir', help='directory containing ground truth parcels raster')
    parser.add_argument('--products_dir', help='directory containing downloaded sentinel products')
    parser.add_argument('--savedir', help='save directory to extract sentinel products windows')
    parser.add_argument('--bands', default=None, help='which satellite image bands to use')
    parser.add_argument('--res', default=10, help='pixel size in meters')
    parser.add_argument('--sample_size', default=24, help='spatial resolution of dataset samples')
    parser.add_argument('--num_processes', default=4, help='number of parallel processes')
    # ---------------------------------------------------------------------------------------------

    args = parser.parse_args()

    ground_truths_dir = args.ground_truths_dir

    products_dir = args.products_dir

    savedir = args.savedir
    print("savedir: ", savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    bands = args.bands
    if bands == 'None':
        bands = list(mult.keys())
    else:
        bands = bands.split(',')

    # res = int(args.res)
    res = float(args.res)

    sample_size = int(args.sample_size)

    num_processes = int(args.num_processes)

    main()
