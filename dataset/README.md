# Process Sentinel products 
What is of interest is to make a timeseries of Sentinel images for the duration of one year given a directory of 
downloaded Sentinel products. The code assumes that all (unziped) Sentinel products saved in the same directory correspond to a 
single tile, for example:
```
products_dir
└───T28PCB
│   └───S2A_MSIL1C_20180702T113321_N0206_R080_T28PCB_20180702T151612.SAFE
│   └───S2A_MSIL1C_20180831T113321_N0206_R080_T28PCB_20180831T153248.SAFE
│   └───...
└───T28PDA
│    └───S2B_MSIL1C_20180316T112109_N0206_R037_T28PDA_20180316T132558.SAFE
│    └───S2B_MSIL1C_20180624T112109_N0206_R037_T28PDA_20180624T132810.SAFE
│    └───...
```

Because the size of a Sentinel tile is too large to fit into gpu memory we split each tile into smaller manageable pieces
of size (HxW) and stack pieces corresponding to the same location at different timestamps to create a timeseries object. 
The final output of the process is a .pickle file with the following contents:
- a numpy array of size (TxH_ixW_i) named after each Sentinel band i. We do not rescale bands to match their resolution 
resulting in a different size for each band. T is the number of available dates
- a numpy array named "doy" of size T which corresponds to the "day of the year" for each available date
- a numpy array named "year" of size 1 corresponding to the year of observations

If ground truth data are available we also include the following:
- a numpy array named "labels" of size (HxW) corresponding to ground truth labels
- a numpy array named "ids" of size (HxW) corresponding to parcel identities

## Including ground truth data
### Make canonical .csv 
If available, ground truth data are used in the form of a canonical .csv file containing the following columns:
- id: (int) object id corresponding to polygon area (optional, if not included a unique integer will be assigned)
- ground_truth: (int) class corresponding to polygon area
- crs: (int) geographic coordinate reference system
- year: (int) the year the ground truth is valid for the given geometry
- geometry: (str) shapely polygon or multipolygon  

For example:
```
ground_truth,crs,year,geometry
1,32628,2019,"POLYGON ((325059.9695234112 1579552.827570891, 325082.9883194482 1579557.590080416, ...))"
2,32628,2019,"POLYGON ((325108.9175379751 1579675.065315364, 325119.871309883 1579667.392383354, .))"
``` 

Specifically for 
[RPG](https://www.data.gouv.fr/en/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire/) 
crop type data for France, the following can be used the following to transform .shp files to canonical .csv:
```shell
python dataset/France_RPG/RPG2DF.py --rpg-dir <RPG files parent directory>
``` 

### Generate data
We distinguish between two different use cases:
1. we overlay a grid of size equal to the desired sample_size on the AOI.
For each grid square we make a raster of all ground truths and satellite images (timeseries) that fall into that square.
The end result is a set of samples of size (sample_size X sample_size) each containing potentially multiple fields and 
not necessarily whole fields as some will be cut at the image boundaries. (show examples) 
2. for each object in the canonical .csv we create a raster ground truth image in which the object is centered and all other pixels not
falling inside the polygon region are assigned the background class. We also generate satellite image timeseries as before. 
This results in a single object per sample at the center of the image.
  
#### Use case 1
For use case 1 run the following bash script to generate data corresponding to spatial locations for which there are available ground 
truths in the form of parcel polygons. 
```shell
sh dataset/labelled_dense/make_labelled_dataset.sh ground_truths_file=<1:ground_truths_file> products_dir=<2:products_dir> labels_dir=<3:labels_dir> windows_dir=<4:windows_dir> timeseries_dir=<5:timeseries_dir> 
res=<6:res> sample_size=<7:sample_size> num_processes<8:num_processes> bands=<8:bands (optional)>
```
where:
- ground_truths_file: file path for canonical .csv file as defined above
- products_dir: directory path for downloaded Sentinel products
- labels_dir: directory to save rasterized ground truths
- windows_dir: directory to save extracted image windows
- timeseries_dir: directory to ave final timeseries objects
- res: highest resolution of satellite image bands, 10 (m) for Sentinel-2
- sample_size: number of pixels of final image windows (for highest resolution image band) and ground truths
- num_processes: number of processes to run on parallel
- bands: (list) which satellite image bands to use, e.g. 'B02,B03,B04,...'. If not specified all bands are used (optional) 

#### Use case 2

For use case 2 we first need to decide on the spatial dimensions of the samples. The following command finds the maximum N-S, E-W 
distance for each parcel as well as the maximum of the two distances and also saves a cummulative histogram for these 
dimensions. 
```shell
python dataset/labelled_dense/find_parcel_dimensions.py ground_truths_file=<1:ground_truths_file> products_dir=<2:products_dir> save_dir=<3:save_dir>
```
where:
- ground_truths_file: file path for canonical .csv file as defined above
- products_dir: directory path for downloaded Sentinel products
- save_dir: directory to save output

This information will help guide the decision on the sample size. We want most parcels to fit in the sample but not make 
it larger than needed as this would mean wasting of computational resources. Of course other considerations may come into play.

After deciding on the parcel size we run the following command to generate use case 2 data:

```shell
sh dataset/labelled_dense/make_labelled_parcel_dataset.sh ground_truths_file=<1:ground_truths_file> products_dir=<2:products_dir> labels_dir=<3:labels_dir> windows_dir=<4:windows_dir> timeseries_dir=<5:timeseries_dir> 
res=<6:res> sample_size=<7:sample_size> Npoly=<8:Npoly> num_processes=<9:num_processes> bands=<10:bands (optional)>
```
```shell
sh dataset/labelled_dense/make_labelled_parcel_dataset.sh ground_truths_file='/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18/example_parcels_in_AOI.csv' products_dir='/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/cloud_0_30' labels_dir='/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18_example/LABELS' windows_dir='/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18_example/IMAGES' timeseries_dir='/media/michaeltrs/sdb/HD2/Data/Satellite_Imagery/RPG/T31FM_18_example/TIMESERIES' res=10 sample_size=100 Npoly=50 num_processes=8
```

## Without ground truth data
In this case we only need to provide the directory where Sentinel products are downloaded. Optionally we can provide an 
anchor point which wil be used when constructing the grid for splitting the AOI into smaller pieces. If provided the 
anchor will be placed at a vertex of the constructed grid. 

### Generate data
Run the following bash script to generate data corresponding to spatial locations for which there are available ground 
truths in the form of parcel polygons. 
```shell
sh dataset/unlabelled/make_unlabelled_dataset.sh products_dir=<1:products_dir> windows_dir=<2:windows_dir> timeseries_dir=<3:timeseries_dir> res=<4:res> 
sample_size=<5:sample_size> num_processes=<6:num_processes> anchor=<7:anchor (optional)> bands=<8:bands (optional)> 
```
where:
- products_dir: (str) directory path for downloaded Sentinel products
- windows_dir: (str) directory to save extracted image windows
- timeseries_dir: (str) directory to ave final timeseries objects
- res: (int) highest resolution of satellite image bands, 10 (m) for Sentinel-2
- sample_size: (int) number of pixels of final image windows (for highest resolution image band) and ground truths
- num_processes: (int) number of processes to run on parallel
- anchor: (list) (N,W,CRS) coordinates of an anchor point and CRS to use as a corner for extracting windows (optional)
- bands: (list) which satellite image bands to use, e.g. 'B02,B03,B04,...'. If not specified all bands are used (optional) 
