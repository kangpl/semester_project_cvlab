# semester_project_cvlab
Semester project at CVLAB

# Dataset preparation
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

# Installation
### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 18.04)
* Python 3.7.4
* PyTorch 1.2.0

### Install semester_project_lab 

a. Clone the semester_project_lab repository.
```shell
git clone https://github.com/kangpl/semester_project_cvlab.git
```

b. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` etc.
```shell
conda create -n myenv python=3.7.4
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch 
conda install scipy numba shapely
conda install -c conda-forge easydict tqdm tensorboardx fire
pip install scikit-image pyyaml
```

c. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
cd PointRCNN
sh build_and_install.sh
```

# PointRCNN (used as baseline)

## Training
Currently, the two stages of PointRCNN are trained separately. First we clarify several terms:
* **GT_AUG** put several new ground-truth boxes and their inside points from other scenes to the same locations of current training scene by randomly selecting non-overlapping boxes, and this augmentation is denoted as GT_AUG. 
* **RCNN online** Train RCNN (2nd stage) network with fixed RPN (1st stage) network
* **RCNN offline** After training the RPN, we save the RoIs and their features for the RCNN training.  
So there are four train strategies:  
(1) with GT_AUG and RCNN online  
(2) with GT_AUG and RCNN offline  
(3) without GT_AUG and RCNN online (**we use this one as our baseline**)  
(4) without GT_AUG and RCNN offline  

### with GT_AUG and RCNN online 
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

### with GT_AUG and RCNN offline 
Step (a)(b) is the same as `with GT_AUG and RCNN online`, except changing the configuration file to `gt_aug_offline_car`. The difference is on step(c)

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
python eval_rcnn.py --cfg_file cfgs/gt_aug_offline_car.yaml --batch_size 4 --eval_mode rpn --ckpt ../output/rpn/gt_aug_offline_car/ckpt/checkpoint_epoch_200.pth --save_rpn_feature
```
3. Now we could train our RCNN network. Note that you should modify `TRAIN.SPLIT=train_aug` to use the augmented scenes for the training, and use `--rcnn_training_roi_dir` and `--rcnn_training_feature_dir` to specify the saved features and proposals in the above step:
```
python train_rcnn.py --cfg_file cfgs/gt_aug_offline_car.yaml --batch_size 4 --train_mode rcnn_offline --epochs 30  --ckpt_save_interval 1 --rcnn_training_roi_dir ../output/rpn/gt_aug_offline_car/eval/epoch_200/train_aug/detections/data --rcnn_training_feature_dir ../output/rpn/gt_aug_offline_car/eval/epoch_200/train_aug/features
```
For the offline GT sampling augmentation, the default setting to train the RCNN network is `RCNN.ROI_SAMPLE_JIT=True`, which means that we sample the RoIs and calculate their GTs in the GPU. I also provide the CPU version proposal sampling, which is implemented in the dataloader, and you could enable this feature by setting `RCNN.ROI_SAMPLE_JIT=False`. Typically the CPU version is faster but costs more CPU resources since they use mutiple workers.  

All the codes supported **mutiple GPUs**, simply add the `--mgpus` argument as above. And you could also increase the `--batch_size` by using multiple GPUs for training.

### without GT_AUG and RCNN online 
The differences between **with GT_AUG and RCNN online** are:
* You don't need to run step(a)
* The configuration file is `PointRCNN/tools/cfgs/no_gt_aug_online_car.yaml`

### without GT_AUG and RCNN offline 
The differences between **with GT_AUG and RCNN offline** are:
* You don't need to run step(a)
* The configuration file is `PointRCNN/tools/cfgs/no_gt_aug_offline_car.yaml`
* You don't need to run step(c)(1)

## Evaluate
* To evaluate a single checkpoint, run the following command with `--ckpt` to specify the checkpoint to be evaluated and `--cfgs_file` to sepecify the corresponding configuration file:
```
python eval_rcnn.py --cfg_file cfgs/gt_aug_online_car.yaml --ckpt ../output/rpn/gt_aug_online_car/ckpt/checkpoint_epoch_200.pth --batch_size 4 --eval_mode rpn 
```

* To evaluate all the checkpoints of a specific training config file, add the `--eval_all` argument, and run the command as follows:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --eval_mode rpn --eval_all
```

* To generate the results on the *test* split, please modify the `TEST.SPLIT=TEST` and add the `--test` argument. 

Here you could specify a bigger `--batch_size` for faster inference based on your GPU memory. Note that the `--eval_mode` argument should be consistent with the `--train_mode` used in the training process. If you are using `--eval_mode=rcnn_offline`, then you should use `--rcnn_eval_roi_dir` and `--rcnn_eval_feature_dir` to specify the saved features and proposals of the validation set. Please refer to the training section for more details. 

## Results
<img src="https://github.com/kangpl/semester_project_cvlab/blob/master/images/baseline_result.png" width="400" height="115">
After comparing different ways of training the PointRCNN, we finally decided to use the training strategy RCNN online without GT_AUG as our baseline. Since this strategy is more elegant and convenient while the performance is also acceptable. Besides, what we want to do is to compare the performance before and after adding the image information.
Here is the pretrained model for these four strategies [pretrained models](https://drive.google.com/drive/folders/1G-eI33TgkPNXdTWEl7SXbkap4RN-qEyh?usp=sharing) from which I get the above results.  


# PointRCNNV1 (add RGB/ add Mean and Covariance)  
<img src="https://github.com/kangpl/semester_project_cvlab/blob/master/images/add_rgb.png" width="500" height="115">
<img src="https://github.com/kangpl/semester_project_cvlab/blob/master/images/add_mean_cov.png" width="500" height="115">

(a) generate rgb/ mean and covariance
* Firstly, get the corresponding `RGB` value for each point cloud using the bilinear interpolation, and get the `mean and covariance value` of a 7X7 patch for each point cloud. Here we will get two files: `train_bgr.pkl` and `train_mean_covariance.pkl` for later usage.
```
python generate_bgr_mean_covariance.py --class_name Car --split train --mode TRAIN
```
* if you want to generate rgb / mean and covariance for evaluation, you nedd to set `--split val`, `--mode EVAL`

(b) Training of RPN stage. 
We can either add `RGB` values or add `mean and covariance` values to each points
* To add the `RGB` values, run the following command:
```
python train_rcnn.py --cfg_file cfgs/use_bgr_car.yaml --batch_size 16 --train_mode rpn --epochs 200 --rpn_bgr '../../data/KITTI/train_bgr.pkl'
```

* To add the `mean and covariance` values, run the following command:
```
python train_rcnn.py --cfg_file cfgs/use_mean_covariance_car.yaml --batch_size 16 --train_mode rpn --epochs 200 --rpn_mean_covariance  '../../data/KITTI/train_mean_covariance.pkl'
```

* If you want to evaluate, run the following command
```
python eval_rcnn.py --cfg_file cfgs/use_bgr_car.yaml --eval_mode rpn --eval_all --batch_size 16 --rpn_bgr '../../data/KITTI/val_bgr.pkl'
```
```
python eval_rcnn.py --cfg_file cfgs/use_mean_covariance_car.yaml --eval_mode rpn --eval_all --batch_size 16 --rpn_mean_covariance '../../data/KITTI/val_mean_covariance.pkl'
```

After training, the checkpoints and training logs will be saved to the corresponding directory according to the name of your configuration file. Such as for the `use_bgr_car.yaml`, you could find the checkpoints and logs in the following directory:
```
PointRCNNV1/output/rpn/use_bgr_car/
```
which will be used for the training of RCNN stage. 

(c) Training of RCNN stage  
Suppose you have a well-trained RPN model saved at `output/rpn/use_bgr_car/ckpt/checkpoint_epoch_200.pth`  
Train RCNN network with fixed RPN network to use online GT augmentation: Use `--rpn_ckpt` to specify the path of a well-trained RPN model and run the command as follows:
* add `RGB` values
```
python train_rcnn.py --cfg_file cfgs/use_bgr_car.yaml --batch_size 4 --train_mode rcnn --epochs 100  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/use_bgr_car/ckpt/checkpoint_epoch_200.pth --rpn_bgr '../../data/KITTI/train_bgr.pkl'
```
* add `mean and covariance` values
```
python train_rcnn.py --cfg_file cfgs/use_mean_covariance_car.yaml --batch_size 4 --train_mode rcnn --epochs 100  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/use_mean_covariance_car/ckpt/checkpoint_epoch_200.pth --rpn_mean_covariance  '../../data/KITTI/train_mean_covariance.pkl'
```

* If you want to evaluate rcnn, run the following command
```
python eval_rcnn.py --cfg_file cfgs/use_bgr_car.yaml --eval_mode rcnn --eval_all --batch_size 4 --rpn_bgr '../../data/KITTI/val_bgr.pkl'
```
```
python eval_rcnn.py --cfg_file cfgs/use_mean_covariance_car.yaml --eval_mode rcnn --eval_all --batch_size 4 --rpn_mean_covariance '../../data/KITTI/val_mean_covariance.pkl'
```

## Results
<img src="https://github.com/kangpl/semester_project_cvlab/blob/master/images/rgb_mean_cov_result.png" width="400" height="115">.   
Here is the pretrained models for [adding rgb](https://drive.google.com/file/d/1q7Bd0EjJ2dGf32uVjs3uLlarC4JqtWAJ/view?usp=sharing) and [adding mean and convariance](https://drive.google.com/file/d/1D5-SUQQTXgU4UxfPeoXJE94oh4q5Lzph/view?usp=sharing) from which I get the above results. You can evaluate the pretained model using the following commands:  
```
python eval_rcnn.py --cfg_file cfgs/use_bgr_car.yaml --ckpt ../../model/add_rgb.pth --batch_size 4 --eval_mode rcnn --rpn_bgr '../../data/KITTI/val_bgr.pkl'
```
```
python eval_rcnn.py --cfg_file cfgs/use_mean_covariance_car.yaml --ckpt ../../model/add_mean_and_cov.pth --batch_size 4 --eval_mode rcnn --rpn_mean_covariance  '../../data/KITTI/val_mean_covariance.pkl'
```

# PointRCNNV2 (add image features to rpn)  
<img src="https://github.com/kangpl/semester_project_cvlab/blob/master/images/add_to_rpn.png" width="500" height="205">

* Please download the finetuned PSPNet model [finetune_car.pth](https://drive.google.com/file/d/1aKYEtYVe0xv_mDGvSorlGU2f5GZy-2jL/view?usp=sharing) and organize the downloaded model as follows: 

```
semester_project_cvlab
├── data 
├── model 
│   ├── finetune_car.pth 
├── PointRCNN 
├── PointRCNNV1 
├── PointRCNNV2 
├── PointRCNNV3 
```

* Training of RPN stage. 
```
python train_rcnn.py --cfg_file cfgs/finetuned_img_features_rpn_car.yaml --batch_size 16 --train_mode rpn --epochs 200
```
* Training of RCNN stage
```
python train_rcnn.py --cfg_file cfgs/finetuned_img_features_rpn_car.yaml --batch_size 4 --train_mode rcnn --epochs 100 --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/finetuned_img_features_rpn_car/ckpt/checkpoint_epoch_200.pth
```

## Results
<img src="https://github.com/kangpl/semester_project_cvlab/blob/master/images/add_image_features_rpn_result.png" width="400" height="105">.   
Here is the pretrained models for [adding image features to rpn](https://drive.google.com/file/d/11RmXBalEPSt410pWPsdw_zDjU_zXQB06/view?usp=sharing) from which I get the above results. You can evaluate the pretained model using the following commands:  
```
python eval_rcnn.py --cfg_file cfgs/finetuned_img_features_rpn_car.yaml --ckpt ../../model/add_img_feature_rpn.pth --batch_size 4 --eval_mode rcnn
```

# PointRCNNV3 (add image features to rcnn)  
<img src="https://github.com/kangpl/semester_project_cvlab/blob/master/images/add_to_rcnn.png" width="500" height="135">

* Please download the finetuned PSPNet model [finetune_car.pth](https://drive.google.com/file/d/1aKYEtYVe0xv_mDGvSorlGU2f5GZy-2jL/view?usp=sharing) and organize it the same as PointRCNNV2

* Training of RPN stage. 
```
python train_rcnn.py --cfg_file cfgs/finetuned_img_feature_rcnn_car.yaml --batch_size 16 --train_mode rpn --epochs 200
```
* Training of RCNN stage
```
python train_rcnn.py --cfg_file cfgs/finetuned_img_feature_rcnn_car.yaml --batch_size 4 --train_mode rcnn --epochs 100 --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/finetuned_img_feature_rcnn_car/ckpt/checkpoint_epoch_200.pth
```

## Results
<img src="https://github.com/kangpl/semester_project_cvlab/blob/master/images/add_image_features_rcnn_result.png" width="400" height="100">.   
Here is the pretrained models for [adding image features to rcnn](https://drive.google.com/file/d/11f_bmNIvCgEt--3qD9g-66gCB3bdL1Eh/view?usp=sharing) from which I get the above results. You can evaluate the pretained model using the following commands:  
```
python eval_rcnn.py --cfg_file cfgs/finetuned_img_feature_rcnn_car.yaml --ckpt ../../model/add_img_feature_rcnn.pth --batch_size 4 --eval_mode rcnn
```
