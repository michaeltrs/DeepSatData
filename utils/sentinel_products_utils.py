import os
from glob import glob
import pandas as pd
import rasterio


def get_S2prod_info(imdirs):
    data = []
    for imdir in imdirs:
        imname = "%s_%s" % (imdir.split("/")[-2].split("_")[-3], imdir.split("/")[-4].split("_")[-5])
        f = rasterio.open("%s/%s_B02.jp2" % (imdir, imname))
        tile_transform = list(f.meta['transform'])
        tile_wn = [tile_transform[2], tile_transform[5]]

        data.append([imdir, imdir.split("/")[-2], imname, tile_wn[0], tile_wn[1], f.meta['height'],
                     f.meta['width'], imname.split("_")[1][:8], f.crs.to_dict()['init']])
    df = pd.DataFrame(data=data,
                      columns=["path", "prod_name1", "prod_name2", "West", "North", "height", "width", "Time",
                               "crs"])  # ,
    # dtype=[np.str, np.str, np.float32, np.float32, np.float32, np.float32, np.str, np.str])
    return df


def get_S2tile_coords(basedir):
    """
    basedir: directory containing sentinel-2 products
    """
    basedir = '/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/cloud_0_30'
    if basedir.split('.')[-1] == 'SAFE':
        imdir = basedir
    elif os.path.dir(basedir):
        files = glob('%s/*.SAFE' % basedir)
        tile = [s.split('/')[-1].split('_')[5] for s in files]
        assert all([t == tile[0] for t in tile]), "not all products in dir correspond to the same tile"
        imdir = files[0]
    imdir = glob("%s/GRANULE/**/IMG_DATA" % imdir)[0]
    # info = get_S2prod_info(filename)
    imname = "%s_%s" % (imdir.split("/")[-2].split("_")[-3], imdir.split("/")[-4].split("_")[-5])
    f = rasterio.open("%s/%s_B02.jp2" % (imdir, imname))
    tile_transform = list(f.meta['transform'])
    tile_wn = [tile_transform[2], tile_transform[5]]
    tile_es = [tile_wn[0] + 10 * f.meta['width'], tile_wn[1] - 10 * f.meta['height']]
    return tile_wn, tile_es


