import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils import data

num_classes = 1
bg_label = 0
root = './datasets/kitti'

# split the train dataset into train and val (150 vs 50)
def make_dataset(mode):
    assert mode in ['train', 'val']
    
    split_dir = os.path.join(root, 'splitIndex', mode + '.txt')
    image_idx_list = [int(x.strip()) for x in open(split_dir).readlines()]
    
    
    img_path = os.path.join(root, 'train', 'image_2')
    gt_path = os.path.join(root, 'train', 'semantic')
    
    dict_id = {-1: bg_label, 0: bg_label, 1: bg_label, 2: bg_label,3: bg_label, 4: bg_label, \
                        5: bg_label, 6: bg_label, 7: bg_label, 8: bg_label, 9: bg_label, 10: bg_label, \
                        11: bg_label, 12: bg_label, 13: bg_label, 14: bg_label, 15: bg_label, 16: bg_label, \
                        17: bg_label, 18: bg_label, 19: bg_label, 20: bg_label, 21: bg_label, 22: bg_label, \
                        23: bg_label, 24: bg_label, 25: bg_label, 26: 1, 27: bg_label, 28: bg_label, \
                        29: bg_label, 30: bg_label, 31: bg_label, 32: bg_label, 33: bg_label}
    
    items = []
    for idx in image_idx_list:
        mask_path = os.path.join(gt_path, "%06d_10.png" % idx)
        mask = Image.open(mask_path)
        mask = np.array(mask)[:360, :1200]
        mask_copy = mask.copy()

        for k, v in dict_id.items():
            mask_copy[mask == k] = v

        if mask_copy.sum() != 0:
            item = (os.path.join(img_path, "%06d_10.png" % idx), os.path.join(gt_path, "%06d_10.png" % idx))
            items.append(item)
    print(len(items))
    return items


class KITTI(data.Dataset):
    def __init__(self,mode = None,joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode = mode)

        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))

        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        # car
        self.id_to_trainid = {-1: bg_label, 0: bg_label, 1: bg_label, 2: bg_label,3: bg_label, 4: bg_label, \
                               5: bg_label, 6: bg_label, 7: bg_label, 8: bg_label, 9: bg_label, 10: bg_label, \
                              11: bg_label, 12: bg_label, 13: bg_label, 14: bg_label, 15: bg_label, 16: bg_label, \
                              17: bg_label, 18: bg_label, 19: bg_label, 20: bg_label, 21: bg_label, 22: bg_label, \
                              23: bg_label, 24: bg_label, 25: bg_label, 26: 1, 27: bg_label, 28: bg_label, \
                              29: bg_label, 30: bg_label, 31: bg_label, 32: bg_label, 33: bg_label}
        # pedestrian
        # self.id_to_trainid = {-1: bg_label, 0: bg_label, 1: bg_label, 2: bg_label,3: bg_label, 4: bg_label, \
        #                 5: bg_label, 6: bg_label, 7: bg_label, 8: bg_label, 9: bg_label, 10: bg_label, \
        #                 11: bg_label, 12: bg_label, 13: bg_label, 14: bg_label, 15: bg_label, 16: bg_label, \
        #                 17: bg_label, 18: bg_label, 19: bg_label, 20: bg_label, 21: bg_label, 22: bg_label, \
        #                 23: bg_label, 24: 1, 25: bg_label, 26: bg_label, 27: bg_label, 28: bg_label, \
        #                 29: bg_label, 30: bg_label, 31: bg_label, 32: bg_label, 33: bg_label}


    def __getitem__(self, index):
        
        img, mask = self.imgs[index]
        img, mask = Image.open(img), Image.open(mask)

        img = np.asarray(img)[:360, :1200]
        img = Image.fromarray(img.astype('uint8'), 'RGB')

        mask = np.array(mask)[:360, :1200]
        mask_copy = mask.copy()

        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8)) 

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            # print(img.shape, mask.shape, torch.LongTensor(slices_info).size, flush=True)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            mask = mask.unsqueeze(0)
            return img, mask
    

    def __len__(self):
        return len(self.imgs)

