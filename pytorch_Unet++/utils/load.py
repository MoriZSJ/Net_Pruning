#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import cv2
import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        img = cv2.imread(dir + id + suffix)
        if img is None:
             img=Image.open(dir + id + '.jpg')
        else:
             img=Image.open(dir + id + suffix)
                
        im = resize_and_crop(img, scale=scale)
        # im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.bmp', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '.bmp', scale)
    masks_normalized = map(normalize, masks)
#     return zip(imgs_normalized, masks)
    return zip(imgs_normalized, masks_normalized)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.bmp')
    mask = Image.open(dir_mask + id + 'bmp')
    return np.array(im), np.array(mask)
