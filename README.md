# DeepSatData: Building large scale datasets of satellite images for training machine learning models
![plot](./diagram.png)
DeepSatData is a toolkit for making datasets from satellite imagery suitable for training machine learning models. 
The process is split into two distinct parts:
- identifying and downloading relevant Sentinel products for an area and time period of interest. Read more in  [download](./download)
- processing downloaded products into datasets. Read more in [dataset](./dataset). 
 
Further details on the methodology used can be found in our papers 
["DeepSatData: Building large scale datasets of satellite images for training machine learning models"](arxiv url) and 
["Context-self contrastive pretraining for crop type semantic segmentation"](https://arxiv.org/abs/2104.04310). 

## Dependencies
Install dependencies using pip
```
pip install -r requirements.txt
```

or creating a conda environment
```
conda create --name <env_name> --file requirements.txt
```

## Citation
If you use DeepSatData in your research consider citing the following BibTeX entries:
```
@misc{tarasiou2021deepsatdata,
      title={DeepSatData: Building large scale datasets of satellite images for training machine learning models}, 
      author={Michail Tarasiou and Stefanos Zafeiriou},
      year={2021},
      eprint={2104.13824},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{tarasiou2021contextself,
      title={Context-self contrastive pretraining for crop type semantic segmentation}, 
      author={Michail Tarasiou and Riza Alp Guler and Stefanos Zafeiriou},
      year={2021},
      eprint={2104.04310},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

<!---## Related projects
This code was used in creating datasets used in [github repo name](github repo url)--->
