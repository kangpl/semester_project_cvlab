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
```shell
conda create -n myenv python=3.7
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch 
conda install scipy numba shapely
conda install -c conda-forge easydict tqdm tensorboardx fire
pip install scikit-image pyyaml
```

c. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```

## PointRCNN (used as baseline)

### Training
Currently, the two stages of PointRCNN are trained separately. First we clarify several terms:
* **GT_AUG** put several new ground-truth boxes and their inside points from other scenes to the same locations of current training scene by randomly selecting non-overlapping boxes, and this augmentation is denoted as GT_AUG. 
* **RCNN online** Train RCNN (2nd stage) network with fixed RPN (1st stage) network
* **RCNN offline** After training the RPN, we save the RoIs and their features for the RCNN training.  
So there are four train strategies:  
(1) with GT_AUG and RCNN online  
(2) with GT_AUG and RCNN offline  
(3) without GT_AUG and RCNN online (we use this one as our baseline)  
(4) without GT_AUG and RCNN offline  

#### with GT_AUG and RCNN online 
(a) generate ground truth
* Firstly, to use the ground truth sampling data augmentation for training, we should generate the ground truth database as follows:
```
python generate_gt_database.py --class_name 'Car' --split train
```
(b) Training of RPN stage
- To train the first proposal generation stage of PointRCNN with a single GPU, run the following command:
```
python train_rcnn.py --cfg_file cfgs/gt_aug_online_car.yaml --batch_size 16 --train_mode rpn --epochs 200
```

- To use **mutiple GPUs for training**, simply add the `--mgpus` argument as follows:
```
CUDA_VISIBLE_DEVICES=0,1 python train_rcnn.py --cfg_file cfgs/gt_aug_online_car.yaml --batch_size 16 --train_mode rpn --epochs 200 --mgpus
```

After training, the checkpoints and training logs will be saved to the corresponding directory according to the name of your configuration file. Such as for the `gt_aug_online_car.yaml`, you could find the checkpoints and logs in the following directory:
```
PointRCNN/output/rpn/gt_aug_online_car/
```
which will be used for the training of RCNN stage. 

(c) Training of RCNN stage  
Suppose you have a well-trained RPN model saved at `output/rpn/gt_aug_online_car/ckpt/checkpoint_epoch_200.pth`  
Train RCNN network with fixed RPN network to use online GT augmentation: Use `--rpn_ckpt` to specify the path of a well-trained RPN model and run the command as follows:
```
python train_rcnn.py --cfg_file cfgs/gt_aug_online_car.yaml --batch_size 4 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/gt_aug_online_car/ckpt/checkpoint_epoch_200.pth
```

#### with GT_AUG and RCNN offline 
Step (a)(b) is the same as `with GT_AUG and RCNN online`, except change the configuration file to `gt_aug_offline_car`. The difference is on step(c)

(c) Train RCNN network with offline GT augmentation: 
1. Generate the augmented offline scenes by running the following command:
```
python generate_aug_scene.py --class_name Car --split train --aug_times 4
```
2. Save the RPN features and proposals by adding `--save_rpn_feature`:

* To save features and proposals for the training, we set `TEST.RPN_POST_NMS_TOP_N=300` and `TEST.RPN_NMS_THRESH=0.85` as follows:
```
python eval_rcnn.py --cfg_file cfgs/gt_aug_offline_car.yaml --batch_size 4 --eval_mode rpn --ckpt ../output/rpn/gt_aug_offline_car/ckpt/checkpoint_epoch_200.pth --save_rpn_feature --set TEST.SPLIT train_aug TEST.RPN_POST_NMS_TOP_N 300 TEST.RPN_NMS_THRESH 0.85
```

* To save features and proposals for the evaluation, we keep `TEST.RPN_POST_NMS_TOP_N=100` and `TEST.RPN_NMS_THRESH=0.8` as default:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --batch_size 4 --eval_mode rpn --ckpt ../output/rpn/gt_aug_offline_car/ckpt/checkpoint_epoch_200.pth --save_rpn_feature
```
3. Now we could train our RCNN network. Note that you should modify `TRAIN.SPLIT=train_aug` to use the augmented scenes for the training, and use `--rcnn_training_roi_dir` and `--rcnn_training_feature_dir` to specify the saved features and proposals in the above step:
```
python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 4 --train_mode rcnn_offline --epochs 30  --ckpt_save_interval 1 --rcnn_training_roi_dir ../output/rpn/default/eval/epoch_200/train_aug/detections/data --rcnn_training_feature_dir ../output/rpn/default/eval/epoch_200/train_aug/features
```
For the offline GT sampling augmentation, the default setting to train the RCNN network is `RCNN.ROI_SAMPLE_JIT=True`, which means that we sample the RoIs and calculate their GTs in the GPU. I also provide the CPU version proposal sampling, which is implemented in the dataloader, and you could enable this feature by setting `RCNN.ROI_SAMPLE_JIT=False`. Typically the CPU version is faster but costs more CPU resources since they use mutiple workers.  

All the codes supported **mutiple GPUs**, simply add the `--mgpus` argument as above. And you could also increase the `--batch_size` by using multiple GPUs for training.

#### with GT_AUG and RCNN online 

#### with GT_AUG and RCNN offline 


### Pretrained model
<img src="https://github.com/kangpl/semester_project_cvlab/blob/master/images/baseline_result.png" width="400" height="115">


### Quick demo
You could run the following command to evaluate the pretrained model (set `RPN.LOC_XZ_FINE=False` since it is a little different with the default configuration): 
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt PointRCNN.pth --batch_size 1 --eval_mode rcnn --set RPN.LOC_XZ_FINE False
```

## Inference
* To evaluate a single checkpoint, run the following command with `--ckpt` to specify the checkpoint to be evaluated:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../output/rpn/ckpt/checkpoint_epoch_200.pth --batch_size 4 --eval_mode rcnn 
```

* To evaluate all the checkpoints of a specific training config file, add the `--eval_all` argument, and run the command as follows:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --eval_mode rcnn --eval_all
```

* To generate the results on the *test* split, please modify the `TEST.SPLIT=TEST` and add the `--test` argument. 

Here you could specify a bigger `--batch_size` for faster inference based on your GPU memory. Note that the `--eval_mode` argument should be consistent with the `--train_mode` used in the training process. If you are using `--eval_mode=rcnn_offline`, then you should use `--rcnn_eval_roi_dir` and `--rcnn_eval_feature_dir` to specify the saved features and proposals of the validation set. Please refer to the training section for more details. 



## PointRCNNV1 (add RGB/ add Mean and Covariance)
## PointRCNNV2 (add image features to rpn)
## PointRCNNV3 (add image features to rcnn)
