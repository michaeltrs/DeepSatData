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
- ground_truth: (int) class id corresponding to polygon area
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
Run the following bash script to generate data corresponding to spatial locations for which there are available ground 
truths in the form of parcel polygons. 
```shell
sh dataset/labelled_dense/make_labelled_dataset.sh <1:ground_truths_file> <2:products_dir> <3:labels_dir> <4:windows_dir> <5:timeseries_dir> 
<6:res> <7:sample_size> <8:num_processes> 
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

## Without ground truth data
In this case we only need to provide the directory where Sentinel products are downloaded. Optionally we can provide an 
anchor point which wil be used when constructing the grid for splitting the AOI into smaller pieces. If provided the 
anchor will be placed at a vertex of the constructed grid. 

### Generate data
Run the following bash script to generate data corresponding to spatial locations for which there are available ground 
truths in the form of parcel polygons. 
```shell
sh dataset/unlabelled/make_unlabelled_dataset.sh <1:products_dir> <2:windows_dir> <3:timeseries_dir> 
<4:res> <5:sample_size> <6:num_processes> <7:anchor (optional)>  
```
where:
- products_dir: (str) directory path for downloaded Sentinel products
- windows_dir: (str) directory to save extracted image windows
- timeseries_dir: (str) directory to ave final timeseries objects
- res: (int) highest resolution of satellite image bands, 10 (m) for Sentinel-2
- sample_size: (int) number of pixels of final image windows (for highest resolution image band) and ground truths
- num_processes: (int) number of processes to run on parallel
- anchor: (list) (N,W,CRS) coordinates of an anchor point and CRS to use as a corner for extracting windows (optional)
