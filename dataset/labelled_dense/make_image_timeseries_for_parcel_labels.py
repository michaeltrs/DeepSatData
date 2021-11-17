"""
For a set of extracted image crops and a labelled_dense label map, make a timeseries of all positions matched with labels
"""
import argparse
import pandas as pd
import numpy as np
import os
import shutil
import pickle
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.insert(0, dir(dir(path[0])))
    __package__ = "examples"
from utils.data_utils import find_number
from utils.date_utils import get_doy
from utils.multiprocessing_utils import run_pool


mult = {'B01': 1/6., 'B02': 1., 'B03': 1., 'B04': 1., 'B05': 1./2., 'B06': 1./2., 'B07': 1./2., 'B08': 1., 'B8A': 1./2,
        'B09': 1./6., 'B10': 1./6., 'B11': 1./2., 'B12': 1./2.}


def match_labels_images(yearlocs):

    refband = bands[1]

    saved_files_info = []
    for jj, yearloc in enumerate(yearlocs):

        if jj % 1000 == 0:
            print("%d of %d" % (jj, len(yearlocs)))
        # yearloc = yearlocs[2100]
        try:

            idx = yearloc_groups[yearloc]
            data = iminfo.iloc[idx, :].sort_values(by='DOY').copy()
            data = data.drop_duplicates(subset=['DOY'], keep='first')  # some products downloaded twice

            Y = data['Year'].iloc[0]
            N = data['Nl'].iloc[0]
            W = data['Wl'].iloc[0]
            # il = data['il'].iloc[0]
            # jl = data['jl'].iloc[0]

            assert all(data['Year'] == Y)
            assert all(data['Nl'] == N)
            assert all(data['Wl'] == W)
            # assert all(data['il'] == il)
            # assert all(data['jl'] == jl)

            # timeseries_sample = {'B01': [], 'B02': [], 'B03': [], 'B04': [], 'B05': [], 'B06': [], 'B07': [],
            #                      'B08': [], 'B8A': [], 'B09': [], 'B10': [], 'B11': [], 'B12': [], 'doy': []}
            timeseries_sample = {band: [] for band in bands}
            timeseries_sample['doy'] = []
            for sample_info in data[['sample_path', 'DOY']].values:
                # sample_info = data[['sample_path', 'DOY']].values[0]
                impath, doy = sample_info

                with open(impath, 'rb') as handle:
                    sample = pickle.load(handle, encoding='latin1')

                # image falls in black region for this product (should have been excluded in extract_images_for_parcel_labels.py)
                if sample[refband].sum() == 0:
                    # print('zero sum')
                    continue

                # image does not match required size (should have been excluded in extract_images_for_parcel_labels.py)
                height, width = sample[refband].shape
                if (height != sample_size) or (width != sample_size):
                    # print('unequal size')
                    continue

                # for key in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']:
                for key in bands:
                    timeseries_sample[key].append(sample[key])
                timeseries_sample['doy'].append(np.array(doy))

            # for key in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'doy']:
            for key in bands:
                timeseries_sample[key] = np.stack(timeseries_sample[key])
            timeseries_sample['doy'] = np.stack(timeseries_sample['doy'])
            timeseries_sample['year'] = np.array(Y).astype(np.int32)

            timesteps = timeseries_sample[refband].shape[0]

            gt = saved_gt_info[(saved_gt_info['Ntl'] == yearloc[0]) & (saved_gt_info['Wtl'] == yearloc[1])]
            with open(gt['filepath'].values[0], 'rb') as handle:
                labels = pickle.load(handle, encoding='latin1')
            for ltype in labels.keys():
                timeseries_sample[ltype.lower()] = labels[ltype]

            savename = os.path.join(year_savedir, "%d_%d_%s.pickle" % (int(N), int(W), Y))
            with open(savename, 'wb') as handle:
                pickle.dump(timeseries_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

            saved_files_info.append([savename, Y, N, W, sample_size, sample_size, timesteps, "completed"])

        except:

            saved_files_info.append(["", Y, N, W, sample_size, sample_size, 0, "failed"])

    saved_files_info = pd.DataFrame(data=saved_files_info, columns=['sample_path', 'Year', 'N', 'W', 'dy', 'dx', 'dt',
                                                                    'status'])
    return saved_files_info


def main():

    global yearloc_groups
    global iminfo
    global labels
    global year_savedir
    global saved_gt_info

    iminfo = pd.read_csv(os.path.join(windows_dir, "extracted_windows_data_info.csv"))
    crs = iminfo['crs'].iloc[0]

    # remove non extracted locations
    iminfo = iminfo[~pd.isnull(iminfo['sample_path'])].reset_index(drop=True)
    iminfo['DOY'] = iminfo['Date'].apply(lambda s: get_doy(str(s)))
    iminfo['Year'] = iminfo['Date'].apply(lambda s: str(s)[:4])

    # ground truths
    # gtinfo = pd.read_csv(os.path.join(windows_dir, "extracted_windows_data_info.csv"))
    #
    # gtfiles = os.listdir(ground_truths_dir)
    # years = [find_number(s, "Y") for s in gtfiles]
    # files = {year: {} for year in set(years)}
    # for i, file in enumerate(gtfiles):
    #     if not file.startswith('INVALID'):
    #         files[years[i]][file.split("_")[0]] = file
    # print("found ground truths in raster for years %s" % ", ".join(list(files.keys())))
    gtfiles = [f for f in os.listdir(ground_truths_dir) if os.path.isdir(os.path.join(ground_truths_dir, f))]

    saved_files_info = []

    for gtfile in gtfiles:
        # gtfile = gtfiles[0]

        saved_gt_info = pd.read_csv(os.path.join(ground_truths_dir, gtfile, 'saved_data_info.csv'))

        year = find_number(gtfile, "Y")
        CRSl = find_number(gtfile, "CRS")

        year_savedir = os.path.join(savedir, year)
        if not os.path.isdir(year_savedir):
            os.makedirs(year_savedir)

        yearloc_groups = iminfo[iminfo['Year'] == year].groupby(['Nl', 'Wl'], as_index=False).groups
        yearlocs = list(yearloc_groups.keys())

        df = run_pool(yearlocs, match_labels_images, num_processes)
        df = pd.concat(df)

        saved_files_info.append(df)


    df = pd.concat(saved_files_info).reset_index(drop=True)
    df['crs'] = crs
    df.to_csv(os.path.join(savedir, "saved_timeseries_data_info.csv"), index=False)

    # delete windows dir
    # shutil.rmtree(windows_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--ground_truths_dir', help='directory containing ground truth parcels raster')
    parser.add_argument('--products_dir', help='directory containing downloaded sentinel products')
    parser.add_argument('--windows_dir', help='directory containing extracted windows from sentinel products')
    parser.add_argument('--savedir', help='save directory for image timeseries with labels')
    parser.add_argument('--bands', default=None, help='which satellite image bands to use')
    parser.add_argument('--res', default=10, help='pixel size in meters')
    parser.add_argument('--sample_size', default=24, help='spatial resolution of dataset samples')
    parser.add_argument('--num_processes', default=4, help='number of parallel processes')
    # ---------------------------------------------------------------------------------------------

    args = parser.parse_args()

    ground_truths_dir = args.ground_truths_dir

    products_dir = args.products_dir

    windows_dir = args.windows_dir

    savedir = args.savedir
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    res = int(args.res)

    sample_size = int(args.sample_size)

    num_processes = int(args.num_processes)

    bands = args.bands

    if bands == 'None':
        bands = list(mult.keys())
    else:
        bands = bands.split(',')

    main()


    # ground_truths_dir = '/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18_3/LABELS'
    # products_dir = '/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/cloud_0_30'
    # windows_dir = '/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18_3/IMAGES'
    # savedir = '/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18_3/TIMESERIES'
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    #
    # res = 10
    # sample_size = 100
    # num_processes = 16
    # bands = 'None'
    #
    #
    # if bands == 'None':
    #     bands = list(mult.keys())
    # else:
    #     bands = bands.split(',')
    #
    # main()
