# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:29:28 2023

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

f = open('stats_metric.json')
data = json.load(f)
data=data['data']
count = 0
class_totals=np.zeros((8))
class_count = np.zeros((8))
for j,dt in enumerate(data):
    if dt['obj']:
        count+=1
    for i,cl in enumerate(dt['class']):
        class_totals[i] += cl
        if cl>0:
            class_count[i] +=1
    if j>300:
        break

class_avg = np.zeros((8)) 
label_map = ["leg:","mouth:","shadow:","tail:","arm:","eye:","body:","background:"]           
for j in range(8):
    class_avg[j] = class_totals[j]/class_count[j]