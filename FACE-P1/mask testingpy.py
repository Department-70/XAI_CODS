# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 08:57:20 2023

@author: Geoffrey Dolinger
"""
import cv2
import os
import json
import numpy as np
from PIL import Image
import matplotlib.colors as color
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours
import tensorflow as tf
from data import test_dataset
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from itertools import compress
from scipy import misc
from model.ResNet_models import Generator
from data_torch import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

blk_image = np.zeros((500,500)) 
   
d_box = [[200,200,300,300],[250,250,350,350],[230,175,320,340],[180,260,370,290]]

for bbox in d_box:
    start=(bbox[0],bbox[1])
    end = (bbox[2],bbox[3])
    box_test_img = cv2.rectangle(blk_image, start, end, (255,0,0), -1)

fig, axis = plt.subplots(1, figsize=(12,6))
axis.imshow(box_test_img)