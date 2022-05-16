# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
from PIL import Image
import cv2

from .seasondepth_mono_dataset import MonoDataset


class SeasonDepthDataset(MonoDataset):
    """Superclass for different types of Season dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(SeasonDepthDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.85, 0, 0.5, 0],
                           [0, 0.69, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        #self.full_res_shape = (1024, 768)  

    def get_color(self, folder, filename, do_flip):
        color = self.loader(self.get_image_path(folder, filename))
        if color.size != (self.width, self.height):
            color = color.resize((self.width, self.height), Image.ANTIALIAS)
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color


class SeasonTrainDataset(SeasonDepthDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(SeasonTrainDataset, self).__init__(*args, **kwargs)
        self.load_depth = False
        
    def get_image_path(self, folder, filename):
        image_path = os.path.join(
            self.data_path, folder, "images", "img_"+filename+"us.jpg")
        return image_path


class SeasonValDataset(SeasonDepthDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(SeasonValDataset, self).__init__(*args, **kwargs)
        self.load_depth = True

    def get_image_path(self, folder, filename):
        image_path = os.path.join(
            self.data_path, "images", folder, "img_"+filename+"us.jpg")
        return image_path

    def get_depth(self, folder, filename):
        gt_path = os.path.join(
            self.data_path, "depth", folder, "img_"+filename+"us.png")
        depth_gt = cv2.imread(gt_path, -1)
        return depth_gt
        
class SeasonTestDataset(SeasonDepthDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(SeasonTestDataset, self).__init__(*args, **kwargs)
        self.load_depth = False

    def get_image_path(self, folder, filename):
        image_path = os.path.join(
            self.data_path, folder, "img_"+filename+"us.jpg")
        return image_path

