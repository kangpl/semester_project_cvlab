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
Here the images are only used for visualization and the [road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing) are optional for data augmentation in the training. 
