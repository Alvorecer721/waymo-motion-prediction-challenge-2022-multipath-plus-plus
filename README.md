# Waymo motion prediction challenge 2022: 3rd place solution (May, 26)
## Our implementation of [MultiPath++](https://arxiv.org/abs/2111.14973)

![](docs/architecture.png)


## General Info:
- 🏎️[**CVPR2022 Workshop on Autonomous Driving website**](https://cvpr2022.wad.vision)
- 📜[**Technical report**](https://arxiv.org/abs/2206.10041)   
- 🥉[**Waymo Motion Prediction Challenge Website**](https://waymo.com/open/challenges/2022/motion-prediction/)

## Team behind this solution:
Stepan Konev 
- [[LinkedIn]](https://www.linkedin.com/in/stepan-konev/)
- [[Twitter]](https://twitter.com/konevsteven)
- [[Facebook]](https://www.facebook.com/stepan.konev.31)

## Code Usage:

For online Python notebook environments the following packages are needed to be installed. Among which the ```nuscenes-devkit``` python library will provide tools to work with the NuScenes dataset. 

``` 
!pip install nuscenes-devkit matplotlib==3.7
import torch
!pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
!pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

Clone the repository ```waymo-motion-prediction-challenge-2022-multipath-plus-plus``` and switch to the specified branch. To be able to import python modules, add the ```code``` directory to the system path. Commands to be run are as follows:

```
!git clone https://github.com/Alvorecer721/waymo-motion-prediction-challenge-2022-multipath-plus-plus multipathpp
!cd multipathpp && git checkout boris/nuscenes-configs

import sys
sys.path.insert(0, '/content/multipathpp/code')
```

First we need to prepare data for training. Data preprocessing is done through rendering objects called visualizers created from a configuration file. The ```MultiPathPPRenderer``` class, which is essentially a visualizer, is utilized to perform prerendering tasks. These tasks involve selection of valid agents and road networks, preperation of road network information such as extracting coordinates of nodes, their types and IDs, transformation of data into agent centric coordinate system, filtering closest road elements, generating embeddings for road segments and trajectory classification of agents basewd on motion predictions. 

The prerendering script will convert the original data format into set of ```.npz``` files each containing the data for a single target agent. From ```code``` folder run
```
!python3 multipathpp/code/prerender/prerender_nuscenes.py \
   --data-version v1.0-mini \
   --data-path drive/MyDrive/multipathpp/nuscenes/v1.0-mini \
   --output-path drive/MyDrive/multipathpp/prerendered_nuscenes \
   --config multipathpp/code/configs/nuscenes_prerender.yaml
```
Rendering is a memory consuming process, therefore it uses multiprocessing to speed up the rendering. So, you may want to use ```n-shards > 1``` and running the script a few times using consecutive ```shard-id``` values

Once we have our data prepared we can run the training.
```
python3 train.py configs/final_RoP_Cov_Single.yaml
```

If you find this work interesting please ⭐️star and share this repo.

## Citation
If you find this work useful please cite us
```
@misc{https://doi.org/10.48550/arxiv.2206.10041,
  doi = {10.48550/ARXIV.2206.10041},
  url = {https://arxiv.org/abs/2206.10041},
  author = {Konev, Stepan},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {MPA: MultiPath++ Based Architecture for Motion Prediction},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
