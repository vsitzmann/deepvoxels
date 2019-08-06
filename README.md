# DeepVoxels

DeepVoxels is an object-specific, persistent 3D feature embedding. It is found by globally optimizing over all available 
2D observations of an object in a deeplearning framework. At test time, the training set can be discarded, and DeepVoxels 
can be used to render novel views of the same object. 

[![deepvoxels_video](https://img.youtube.com/vi/HM_WsZhoGXw/0.jpg)](https://www.youtube.com/watch?v=-Vto65Yxt8s)

## Usage
### Installation
This code was developed in python 3.7 and pytorch 1.0. I recommend to use anaconda for dependency management. 
You can create an environment with name "deepvoxels" with all dependencies like so:
```
conda env create -f environment.yml
```

### High-Level structure
The code is organized as follows:
* dataio.py loads training and testing data.
* data_util.py and util.py contain utility functions.
* run_deepvoxels.py contains the training and testing code as well as setting up the dataset, dataloading, command line arguments etc.
* deep_voxels.py contains the core DeepVoxels model.
* custom_layers.py contains implementations of the integration and occlusion submodules.
* projection.py contains utility functions for 3D and projective geometry.

### Data
The datasets have been rendered from a set of high-quality 3D scans of a variety of objects.
The datasets are available for download [here](https://drive.google.com/open?id=1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH).
Each object has its own directory, which is the directory that the "data_root" command-line argument of the run_deepvoxels.py script is pointed to.

## Coordinate and camera parameter conventions
This code uses an "OpenCV" style camera coordinate system, where the Y-axis points downwards (the up-vector points in the negative Y-direction), 
the X-axis points right, and the Z-axis points into the image plane. Camera poses are assumed to be in a "camera2world" format,
i.e., they denote the matrix transform that transforms camera coordinates to world coordinates.

The code also reads an "intrinsics.txt" file from the dataset directory. This file is expected to be structured as follows:
```
f cx cy
origin_x origin_y origin_z
near_plane (if 0, defaults to sqrt(3)/2)
scale
img_height img_width
```
The focal length, cx and cy are in pixels. (origin_x, origin_y, origin_z) denotes the origin of the voxel grid in world coordinates.
The near plane is also expressed in world units. Per default, each voxel has a sidelength of 1 in world units - the scale is a 
factor that scales the sidelength of each voxel. Finally, height and width are the resolution of the image.

To create your own dataset, I recommend using the amazing, open-source [Colmap](https://colmap.github.io/install.html). Follow the instructions on the website to install it.
I have written a little wrapper in python that will automatically reconstruct a directory of images, and then
extract the camera extrinsic & intrinsic camera parameters. It can be used like so:
```
python colmap_wrapper.py --img_dir [path to directory with images] \
                         --trgt_dir [path where output will be written to] 
```
To get the scale and origin of the voxel grid as well as the near plane, one has to inspect the reconstructed point cloud and manually
edit the intrinsics.txt file written out by colmap_wrapper.py.

### Training
* See `python run_deepvoxels.py --help` for all train options. 
Example train call:
```
python run_deepvoxels.py --train_test train \
                         --data_root [path to directory with dataset] \
                         --logging_root [path to directory where tensorboard summaries and checkpoints should be written to] 
```
To monitor progress, the training code writes tensorboard summaries every 100 steps into a "runs" subdirectory in the logging_root.

### Testing
Example test call:
```
python run_deepvoxels.py --train_test test \
                         --data_root [path to directory with dataset] ]
                         --logging_root [path to directoy where test output should be written to] \
                         --checkpoint [path to checkpoint]
```

## Misc
### Citation:  
If you find our work useful in your research, please consider citing:
```
@inproceedings{sitzmann2019deepvoxels,
	author = {Sitzmann, Vincent 
	          and Thies, Justus 
	          and Heide, Felix 
	          and Nie{\ss}ner, Matthias 
	          and Wetzstein, Gordon 
	          and Zollh{\"o}fer, Michael},
	title = {DeepVoxels: Learning Persistent 3D Feature Embeddings},
	booktitle = {Proc. CVPR},
	year={2019}
}
```

### Follow-up work
Check out our new project, [Scene Representation Networks](https://vsitzmann.github.io/srns/), where we 
replace the voxel grid with a continuous function that naturally generalizes across scenes and smoothly parameterizes scene surfaces!

### Submodule "pytorch_prototyping"
The code in the subdirectory "pytorch_prototyping" comes from a little library of custom pytorch modules that I use throughout my 
research projects. You can find it [here](https://github.com/vsitzmann/pytorch_prototyping).

### Other cool projects
Some of the code in this project is based on code from these two very cool papers:
* [Learning a Multi-View Stereo Machine](https://github.com/akar43/lsm)
* [3DMV](https://github.com/angeladai/3DMV)

Check them out!

### Contact:
If you have any questions, please email Vincent Sitzmann at sitzmann@cs.stanford.edu.
