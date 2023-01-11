import numpy as np
import torch
import random
import cv2
from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform

class CutMix(T.Augmentation):
    def get_transform(self, image1, image2, label1, label2):
    	pass

class CutOutPolicy(T.Augmentation):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.min_holes, self.max_holes = n_holes
        self.min_length, self.max_length = length

    def get_transform(self, image):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = image.shape[0]
        w = image.shape[1]
        holes = np.random.randint(self.min_holes, high=self.max_holes)
        mask = np.ones(image.shape, np.float32)

        for n in range(holes):
            xlength = np.random.randint(self.min_length, high=self.max_length)
            ylength = np.random.randint(self.min_length, high=self.max_length)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - ylength // 2, 0, h)
            y2 = np.clip(y + ylength // 2, 0, h)
            x1 = np.clip(x - xlength // 2, 0, w)
            x2 = np.clip(x + xlength // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        return CutOut(mask)

class CutOut(Transform):
    def __init__(self, mask):
        self.mask = mask        

    def apply_image(self, img, seg_mode=False):
        if seg_mode:
            img = img * self.mask[:,:,0]
            img = img + ((1-self.mask[:,:,0])*200) # CutOut pixels set to 200 to detect them later and create a mask for the loss
        else:
            img = img * self.mask
        return img

    def apply_segmentation(self, segmentation):
        return segmentation #self.apply_image(segmentation, seg_mode=True)

    def apply_coords(self, coords):
        return coords

class TrainScalePolicy(T.Augmentation):
    def __init__(self, train_scale):
        self.lscale, self.hscale = train_scale

    def get_transform(self, image):
        f_scale = self.lscale + random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
        return TrainScale(f_scale)

class TrainScale(Transform):
    def __init__(self, f_scale):
        self.f_scale = f_scale

    def apply_image(self, image):
        image = cv2.resize(image, None, fx=self.f_scale, fy=self.f_scale, interpolation = cv2.INTER_LINEAR)
        return image

    def apply_segmentation(self, segmentation):
        segmentation = cv2.resize(segmentation, None, fx=self.f_scale, fy=self.f_scale, interpolation = cv2.INTER_NEAREST)
        return segmentation

    def apply_coords(self, coords):
        return coords


