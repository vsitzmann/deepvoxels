# DeepVoxels

DeepVoxels is an object-specific, persistent 3D feature embedding. It is found by globally optimizing over all available 
2D observations of an object in a deeplearning framework. At test time, the training set can be discarded, and DeepVoxels 
can be used to render novel views of the same object. 

[![deepvoxels_video](https://img.youtube.com/vi/HM_WsZhoGXw/0.jpg)](https://www.youtube.com/watch?v=HM_WsZhoGXw)

## Usage
### Installation
This code was developed in python 3.7 and pytorch 1.0. I recommend to use anaconda for dependency management. 
You can create an environment with name "deepvoxels" with all dependencies like so:
```
conda env create -f src/environment.yml
```

### Training:  
* See `python run_deepvoxels.py --help` for all train options. 
Example train call:
```
python run_deepvoxels.py --train_test train \
                         --data_root [path to directory with dataset] \
                         --logging_root [path to directory where tensorboard summaries and checkpoints should be written to] 
```
To monitor progress, the training code writes tensorboard summaries every 100 steps into a "runs" subdirectory in the logging_root.

The default settings in this repository are slightly different than those discussed in the paper: We set the grid 
sidelength (grid_dim) to 64 and reduce the number of features (num_grid_feats) to 8. This yields results that are 
superior to those reported in the paper. If you want to stay true to the paper, set grid_dim to 32 and num_grid_feats to 64.

### Testing
Example test call:
```
python run_deepvoxels.py --train_test test \
                         --data_root [path to directory with dataset] ]
                         --logging_root [path to directoy where test output should be written to] \
                         --checkpoint [path to checkpoint]
```

### Data:
The datasets have been rendered from a set of high-quality 3D scans of a variety of objects.
The dataset is available for download [here](https://drive.google.com/open?id=1KlWzDdKpmSZ-J-PZFp7j8Z9tUHUvTsE-).

### Trained models
I've trained models with the default settings for the four objects. The checkpoints can be found 
[here](https://drive.google.com/drive/folders/1EpMJZPLFVDwXFIvrQ4QSUc9DIXqX67cx?usp=sharing).

## Misc
### Citation:  
If you find our work useful in your research, please consider citing:
```
@article{sitzmann2018deepvoxels,
  title={DeepVoxels: Learning Persistent 3D Feature Embeddings},
  author={Sitzmann, Vincent 
          and Thies, Justus 
          and Heide, Felix 
          and Nie{\ss}ner, Matthias 
          and Wetzstein, Gordon 
          and Zollh{\"o}fer, Michael},
  journal={arXiv preprint arXiv:1812.01024},
  year={2018}
}
```

### Submodule "pytorch_prototyping"
The code in the subdirectory "pytorch_prototyping" comes from a little library of custom pytorch modules that I use throughout my 
research projects. You can find it [here](https://github.com/vsitzmann/pytorch_prototyping).

### Other cool projects
Some of the code in this project is based on code from these two very cool papers:
[Learning a Multi-View Stereo Machine](https://github.com/akar43/lsm)
[3DMV](https://github.com/angeladai/3DMV)
Check them out!

### Future work
We have more cool work on novel view synthesis, neural scene representations and neural rendering in the pipeline - 
stay tuned!

### Contact:
If you have any questions, please email Vincent Sitzmann at sitzmann@cs.stanford.edu.
