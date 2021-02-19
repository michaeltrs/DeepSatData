# Data download

## General Description
We propose to split the task of downloading all relevant products to an AOI during a specific time period to the 
following subtasks:
1) for an AOI find all overlaping Sentinel tiles using **`find_tiles_for_aoi.ipynb`**
2) make a list of all products to download for each tile and period of interest and save a file with all selected 
products to disk using **`find_products_for_tile.ipynb`**. During this step some compromises might be needed to reduce 
the total download time.
3) download all selected products with **`aoi_download.sh`**

Downloading data can take a significant amount of time. 
We propose to perform steps 1,2 manually using the provided .ipynb files to ensure an optimal selection of products and 
automate the final part of downloading a list of pre-selected products. 

## Authentication
All scripts make use of the [sentinelsat]{https://github.com/sentinelsat/sentinelsat} library for querying and 
downloading Sentinel products from the ESA [Copernicus Open Access Hub]{https://scihub.copernicus.eu/} (COAH). 
You will need to [sign up]{https://scihub.copernicus.eu/dhus/#/self-registration} to COAH and save the user name and 
password in a two row file **pw.csv** with the following form:
```
username
password
``` 

### Find Sentinel tiles for AOI
Notebook **`find_tiles_for_aoi.ipynb`**
- AOI is defined as a rectangle. Define the coordinates of the North-West (NW) and South-East (SE) corners of the AOI 
as well as coordinate system used
- output is all tiles that overlap with defined rectangle and part of the area of teh rectangle covered by each tile. 
Note that for very large AOIs all tiles will cover a small portion of the defined rectangle

### Find a list of products 
Notebook **find_products_for_tile.ipynb**
- specify the following parameters:
    - savedir: where to save products list
    - year: 'yyyy'
    - date_range: minimum and maximum dates for the same year (mindate: 'mmdd', maxdate: 'mmdd')
    - cloudcoverpercentage: minimum and maximum percentage of cloud cover (min %, max %)
    - minprodsize: minimum size of product in Mb
    - numproducts: number of products to select 
    - tile: Sentinel tile name e.g. '35PNK' from previous step
    - platformname: Sentinel mission name i.e. 'Sentinel-2'
    - processinglevel: processing level of products i.e. 'Level-1C'
- the script queries COAH for available products given the parameters set and selects products such that they are 
equally spaced in the defined time period. All selected products are saved in a .csv file.

### Download data from file
Pass one or more generated .csv files containing selected products to **download.sh** separated by commas:
```
sh download/download.sh file1.csv,file2.csv,...
``` 
he products will be downloaded in the parent directory of the first .csv file.

## TODO
- [] in get tiles file plot geo of tiles and aoi
