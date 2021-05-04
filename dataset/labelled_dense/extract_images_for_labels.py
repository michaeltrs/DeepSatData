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


mult = {'B01': 1/6.,'B02': 1., 'B03': 1., 'B04': 1., 'B05': 1./2., 'B06': 1./2., 'B07': 1./2., 'B08': 1., 'B8A': 1./2,
        'B09': 1./6., 'B10': 1./6., 'B11': 1./2., 'B12': 1./2.}
# jp2s = ["%s.jp2" % i for i in mult.keys()]


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

        if str(f.crs).split(':')[1] != CRSl:
            geotransform_prod2label = GeoTransform(str(f.crs).split(':')[1], CRSl, loc2loc=CRSl != '4326')
            Wp, Np = geotransform_prod2label(np.array(f.transform)[2], np.array(f.transform)[5])
        else:
            Wp, Np = np.array(f.transform)[2], np.array(f.transform)[5]

        prod_savedir = os.path.join(savedir, imdir.split("/")[-4].split(".")[0])
        if not os.path.exists(prod_savedir):
            os.makedirs(prod_savedir)

        for i in range(int(num_rows)):

            for j in range(int(num_cols)):

                if i * num_cols + j == 1000:
                    print("row %d of %d, column %d of %d" % (i, num_rows, j, num_cols))

                Nij = Nl - i * res * sample_size  # N for extracted label window
                Wij = Wl + j * res * sample_size  # W for extracted label window
                ip = (Np - Nij) / (res * sample_size)  # product row
                jp = (Wij - Wp) / (res * sample_size)  # product column

                if (ip < 0) or (jp < 0):
                    saved_files_info.append(
                        [None, Nij, Wij, Nl, Wl, Np, Wp, i, j, ip, jp, sample_size, sample_size, date, imdir,
                         "sample outside Sentinel product"])
                    continue

                date = imdir.split("/")[-4].split(".")[0].split("_")[2][:8]

                if labels[i * sample_size: (i + 1) * sample_size, j * sample_size: (j + 1) * sample_size].sum() == 0:
                    saved_files_info.append(
                        [None, Nij, Wij, Nl, Wl, Np, Wp, i, j, ip, jp, sample_size, sample_size, date, imdir, "no labels"])
                    continue

                sample = {}
                for jp2 in jp2s:
                    xpmin = int(np.round(mult[jp2[:-4]] * ip * sample_size))
                    ypmin = int(np.round(mult[jp2[:-4]] * jp * sample_size))
                    sample[jp2[:-4]] = data[jp2[:-4]][xpmin: xpmin + int(mult[jp2[:-4]] * sample_size),
                                       ypmin: ypmin + int(mult[jp2[:-4]] * sample_size)]

                if sample[jp2[:-4]].sum() == 0:
                    saved_files_info.append(
                        [None, Nij, Wij, Nl, Wl, Np, Wp, i, j, ip, jp, sample_size, sample_size, date, imdir, "no image"])
                    continue

                sample_save_path = "%s/N%d_W%d_D%s.pickle" % (prod_savedir, int(Nij), int(Wij), date)
                with open(sample_save_path, 'wb') as handle:
                    pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

                saved_files_info.append(
                    [sample_save_path, Nij, Wij, Nl, Wl, Np, Wp, i, j, ip, jp, sample_size, sample_size, date, imdir, "ok"])

    df = pd.DataFrame(data=saved_files_info,
                      columns=['sample_path', 'Nij', 'Wij', 'Nl', 'Wl', 'Np', 'Wp', 'il', 'jl', 'ip', 'jp',
                               'height', 'width', 'Date', 'S2_prod_imdir', "comment"])
    return df


def main():
    # ground truths
    gtfiles = os.listdir(ground_truths_dir)
    years = [find_number(s, "Y") for s in gtfiles]
    files = {year: {} for year in set(years)}
    for i, file in enumerate(gtfiles):
        if not file.startswith('INVALID'):
            files[years[i]][file.split("_")[0]] = file
    print("found ground truths in raster for years %s" % ", ".join(list(files.keys())))
    global labels
    global Nl
    global Wl
    global CRSl
    global num_rows
    global num_cols

    # sentinel products
    imdirs = glob("%s/*.SAFE/GRANULE/**/IMG_DATA" % products_dir)
    prod_df = get_S2prod_info(imdirs)
    prod_df['Year'] = prod_df['Time'].apply(lambda s: s[:4])

    out = []
    for year in set(years):

        # ground truths
        labels = np.loadtxt(os.path.join(ground_truths_dir, files[year]['LABELS']), dtype=np.float32)
        Nl = int(find_number(files[year]['LABELS'], "N"))
        Wl = int(find_number(files[year]['LABELS'], "W"))

        num_rows, num_cols = [d / sample_size for d in labels.shape]
        assert (np.ceil(num_rows) == num_rows) and (np.ceil(num_cols) == num_cols), \
        "sample size should be fitting exactly in labels, this suggests an error in extract_labels_raster script"
        CRSl = find_number(files[year]['LABELS'], "CRS")

        # sentinel products
        products = prod_df[prod_df['Year'] == year]
        imdirs = products['path'].tolist()

        df_year = run_pool(imdirs, extract_images, num_processes)
        # df = extract_images([imdirs[0]])
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

    res = int(args.res)

    sample_size = int(args.sample_size)

    num_processes = int(args.num_processes)

    main()
