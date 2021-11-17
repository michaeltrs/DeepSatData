# spatial data processing pipelines
import argparse
import pandas as pd
from sentinelsat import SentinelAPI  # , read_geojson, geojson_to_wkt
import os
from glob import glob
from collections import OrderedDict


# USER INPUT -----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--products_file', metavar='PRODUCTS FILE', default='',
                    help='path to file containing all products to be downloaded')

args = parser.parse_args()
products_file = args.products_file

# CODE -----------------------------------------------------------------------------------------------------------------
# authentication
cred = pd.read_csv("download/pw.csv", header=None)
api = SentinelAPI(cred[0][0], cred[0][1], 'https://apihub.copernicus.eu/apihub')  # 'https://scihub.copernicus.eu/dhus')  #

# read products to download from file
if ',' in products_file:
    products_file = products_file.split(',')
    savedir = os.path.dirname(products_file[0])
    products = pd.concat([pd.read_csv(products_file_) for products_file_ in products_file])
else:
    savedir = os.path.dirname(products_file)
    products = pd.read_csv(products_file)

# make products into ordered dict
products2download = OrderedDict()
for i in range(products.shape[0]):  # enumerate(list(products.keys())):
    products2download[products['index'].iloc[i]] = products.iloc[i].to_dict()

# find number of remaining products
down_filenames = [os.path.basename(p).split(".")[0] for p in glob(os.path.join(savedir, "*.zip"))]
N = 0
for key in products2download:
    if products2download[key]['identifier'] in down_filenames:
        N += 1
print("%d of %d new products already downloaded, %d remaining" % (N, len(products2download), len(products2download)-N))

# download
# try:
api.download_all(products2download, directory_path=savedir, n_concurrent_dl=1)
# except:
#     p
print("waiting 30min...")
