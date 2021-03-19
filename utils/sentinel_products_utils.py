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
