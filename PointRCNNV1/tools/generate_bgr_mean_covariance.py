import _init_path
import os
import numpy as np
import pickle
import torch

import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.datasets.kitti_dataset import KittiDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--class_name', type=str, default='Car')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--mode', type=str, default='TRAIN')
args = parser.parse_args()
INCLUDE_SIMILAR_TYPE = True
PC_REDUCE_BY_RANGE =  True
PC_AREA_SCOPE = [[-40, 40], [-1,   3], [0, 70.4]]  # x, y, z scope in rect camera coords

class MeanCovGenerator(KittiDataset):
    def __init__(self, root_dir, split='train', classes='Car', mode='TRAIN'): #split shoule be "train" or "val"
        super().__init__(root_dir=root_dir, split=split)
        if classes == 'Car':
            self.classes = ('Background', 'Car')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode
        self.sample_id_list = []
        self.root_dir = root_dir
        self.split = split

        if mode == 'TRAIN':
            self.preprocess_rpn_training_data()
        else: #if moed is EVAL do not need to filter
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
            print('Load testing samples from %s' % self.imageset_dir)
            print('Done: total test samples %d' % len(self.sample_id_list))

    def generate_mean_covariance(self):
        # store mean_covariance value before training
        self.bgr = []
        self.mean_covariance = []

        print(">>>>>>>>>>>>>>>>>>>>>>>>start preprocessing data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        for x in self.sample_id_list:
            sample_id = int(x)
            print("start processing sample: ", sample_id)
            calib = self.get_calib(sample_id)
            img = self.get_image(sample_id)
            img_shape = self.get_image_shape(sample_id)
            pts_lidar = self.get_lidar(sample_id)

            # get valid point (projected points should be in image)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
            pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

            pts_rect = pts_rect[pts_valid_flag][:, 0:3]
            self.bgr.append( self.get_bgr(img, pts_img[pts_valid_flag]) )#(n, 3) each with bgr value
            self.mean_covariance.append(self.get_mean_covariance(img, pts_img[pts_valid_flag]))

        save_file_name = os.path.join(self.root_dir, 'KITTI', '%s_bgr.pkl' % (self.split))
        with open(save_file_name, 'wb') as f:
            pickle.dump(self.bgr, f)
        print('Save bgr sample info file to %s' % save_file_name)
        save_file_name = os.path.join(self.root_dir, 'KITTI', '%s_mean_covariance.pkl' % (self.split))
        with open(save_file_name, 'wb') as f:
            pickle.dump(self.mean_covariance, f)
        print('Save mean covariance sample info file to %s' % save_file_name)
        print("<<<<<<<<<<<<<<<<<<<<<<<<finished preprocessing data<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def preprocess_rpn_training_data(self):
        """
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        print('Loading %s samples from %s ...' % (self.mode, self.label_dir))
        for idx in range(0, self.num_sample):
            sample_id = int(self.image_idx_list[idx])
            obj_list = self.filtrate_objects(self.get_label(sample_id))
            if len(obj_list) == 0:
                # self.logger.info('No gt classes: %06d' % sample_id)
                continue
            self.sample_id_list.append(sample_id)

        print('Done: filter %s results: %d / %d\n' % (self.mode, len(self.sample_id_list),
                                                                 len(self.image_idx_list)))

    def get_label(self, idx):
        return super().get_label(idx % 10000)

    def get_image(self, idx):
        return super().get_image(idx % 10000)

    def get_image_shape(self, idx):
        return super().get_image_shape(idx % 10000)

    def get_calib(self, idx):
        return super().get_calib(idx % 10000)

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        if self.mode == 'TRAIN' and INCLUDE_SIMILAR_TYPE:
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
            if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
                type_whitelist.append('Person_sitting')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:  # rm Van, 20180928
                continue
            if self.mode == 'TRAIN' and PC_REDUCE_BY_RANGE and (self.check_pc_range(obj.pos) is False):
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    @staticmethod
    def check_pc_range(xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = PC_AREA_SCOPE
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag
    
    @staticmethod
    def get_bgr(img, pts_img):
        """
        Use bilinear interpolation to get the bgr value for each Lidar points
        :param img: 2d image
        :param pts_img: valid lidar points on image2 coord 
        :return: (n, 3)  bgr mode
        """
        x = pts_img[:, 0]
        y = pts_img[:, 1]

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, img.shape[1]-1)
        x1 = np.clip(x1, 0, img.shape[1]-1)
        y0 = np.clip(y0, 0, img.shape[0]-1)
        y1 = np.clip(y1, 0, img.shape[0]-1)

        Ia = img[ y0, x0 ]
        Ib = img[ y1, x0 ]
        Ic = img[ y0, x1 ]
        Id = img[ y1, x1 ]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T

    @staticmethod
    def get_mean_covariance(img, pts_img):
        """
        Use 7*7 patch to get the bgr value and covariance vector for each Lidar points
        :param img: 2d image
        :param pts_img: valid lidar points on image2 coord 
        :return: (n, 9)  b g r covariance of bb bg br gg gr rr 
        """
        x = pts_img[:, 0]
        y = pts_img[:, 1]

        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        
        x_min = np.clip(x0-3, 0, img.shape[1]-1)
        x_max = np.clip(x0+4, 0, img.shape[1]-1)
        y_min = np.clip(y0-3, 0, img.shape[0]-1)
        y_max = np.clip(y0+4, 0, img.shape[0]-1)
        
        mean_list = []
        cov_list = []
        for (ya, yb, xa, xb) in zip(y_min, y_max, x_min, x_max):
            patch = img[ya:yb, xa:xb]
            mean_list.append(patch.mean(axis=(0,1)))          
            cov_list.append(np.cov(patch.reshape(-1,3).T)[np.triu_indices(3)])
        mean = np.asarray(mean_list)
        cov_vector = np.asarray(cov_list)
        mean_covariance = np.concatenate((mean, cov_vector), axis=1)
        
        return mean_covariance

if __name__ == '__main__':
    dataset = MeanCovGenerator(root_dir='../../data/', split=args.split, mode=args.mode)

    dataset.generate_mean_covariance()

    # bgr_file = '../data/KITTI/train_bgr.pkl'
    # mean_covariance_file = '../data/KITTI/train_mean_covariance.pkl'
    # bgr = pickle.load(open(bgr_file, 'rb'))
    # mean_covariance = pickle.load(open(mean_covariance_file, 'rb'))
    # print('Loading bgr(%d) from %s' % (len(bgr), bgr_file))
    # print('Loading mean_covariance(%d) from %s' % (len(mean_covariance), mean_covariance_file))

