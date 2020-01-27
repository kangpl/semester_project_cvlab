# semester_project_cvlab
Semester project at CVLAB

## Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows: 
```
data
├── KITTI
│   ├── ImageSets
│   ├── object
│   │   ├──training
│   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   ├──testing
│   │      ├──calib & velodyne & image_2
```
Here the [road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing) are optional for ground truth augmentation in the training. 

## Installation
### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 18.04)
* Python 3.6+
* PyTorch 1.0+

### Install semester_project_lab 

a. Clone the semester_project_lab repository.
```shell
git clone https://github.com/kangpl/semester_project_cvlab.git
```

b. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` etc.

c. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```
