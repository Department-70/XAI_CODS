# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 20:11:09 2023

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

def calc_area(box):
    return (box[2]-box[0])*(box[3]-box[1])

def overlap(B1, B2):
    if (B1[0]>=B2[2]) or (B1[2]<=B2[0]) or (B1[3]<=B2[1]) or (B1[1]>=B2[3]):
        return False
    else:
        return True

def outside_box(box, test_box):
    out=[]
    pts = [[test_box[0],test_box[1]],[test_box[2],test_box[1]],[test_box[0],test_box[3]],[test_box[2],test_box[3]]]
    for p in pts:
        out.append(not(p[0] >= box[0] and p[0] <= box[2] and p[1] >= box[1] and p[1] <= box[3])) 
    return out, pts         

def box_2pts(box, pts):
    if pts[0][0]< box[0]:
        boxes = [box,[pts[0][0],pts[0][1],box[0],pts[1][1]]]
    elif pts[0][0]> box[2]:
        boxes = [box,[box[2],pts[0][1],pts[1][0],pts[1][1]]]
    elif pts[0][1]< box[1]:
        boxes = [box,[pts[0][0],pts[0][1],pts[1][0],box[1]]]
    else:
        boxes = [box,[pts[0][0],box[3],pts[1][0],pts[1][1]]]
    return boxes

def box_intersect(box, pts):
    if pts[0][0] < box[0]:
        boxes = [box, [ pts[0][0], pts[0][1], box[0], pts[2][1] ],
                      [ box[2], pts[0][1], pts[3][0], pts[3][1] ] ] 
    else:
        boxes = [box, [ pts[0][0], pts[0][1], pts[1][0], box[1] ],
                      [ pts[0][0], box[3], pts[3][0], pts[3][1] ] ]
    return boxes
        
def box_3pts(box, pts):
    if pts[0][0] < box[0]:
        if pts[0][1]< box[1]:
            if abs(pts[0][0]-pts[1][0])>abs(pts[0][1]-pts[2][1]):
                boxes = [box, [ pts[0][0], pts[0][1], pts[1][0], box[1] ],
                              [ pts[0][0], box[1], box[0], pts[2][1] ] ]
            else:
                boxes = [box, [ pts[0][0], pts[0][1], box[0], pts[2][1] ],
                              [ box[0], pts[1][1], pts[1][0], box[1] ] ]
        else:
            if abs(pts[0][0]-pts[2][0])>abs(pts[0][1]-pts[1][1]):
                boxes = [box, [ pts[0][0], box[3], pts[2][0], pts[2][1] ],
                              [ pts[0][0], pts[0][1], box[0], box[3] ] ]
            else:
                boxes = [box, [ pts[0][0], pts[0][1], box[0], pts[1][1] ],
                              [ box[0], box[3], pts[2][0], pts[2][1] ] ]
    elif pts[0][1] < box[1]:
        if abs(pts[0][0]-pts[1][0])>abs(pts[0][1]-pts[2][1]):
            boxes = [box, [ pts[0][0], pts[0][1], pts[1][0], box[1] ],
                          [ box[2], box[1], pts[2][0], pts[2][1] ] ]
        else:
            boxes = [box, [ box[2], pts[0][1], pts[2][0], pts[2][1] ],
                          [ pts[0][0], pts[0][1], box[2], box[1] ] ]
    else:
        if abs(pts[0][0]-pts[1][0])>abs(pts[0][1]-pts[2][1]):
            boxes = [box, [ pts[1][0], box[3], pts[2][0], pts[2][1] ],
                          [ box[2], pts[0][1], pts[2][0], box[3] ] ]
        else:
            boxes = [box, [ box[2], pts[0][1], pts[2][0], pts[2][1] ],
                          [ pts[1][0], box[3], box[2], pts[2][1] ] ]
    return boxes

def box_split(B1,B2):
    ovlap = overlap(B1, B2) 
    out12, B2_pts = outside_box(B1,B2)
    out21, B1_pts = outside_box(B2,B1)
    if not ovlap:
        box_out = [B1,B2]
    elif sum(out12)>3 and sum(out21)>3: 
        if calc_area(B1)>=calc_area(B2):
            mainB = B1
            sec_pts = list(compress(B2_pts,out12))
        else:
            mainB = B2
            sec_pts = list(compress(B1_pts,out21))
        box_out = box_intersect(mainB,sec_pts)
    elif sum(out12)==2 or sum(out21)==2:
        if sum(out12)==2:
            mainB = B1
            sec_pts = list(compress(B2_pts,out12))
        else:
            mainB = B2
            sec_pts = list(compress(B1_pts,out21))
        box_out = box_2pts(mainB,sec_pts)
    elif sum(out12)==3 or sum(out21)==3:
        if calc_area(B1)>=calc_area(B2):
            mainB = B1
            sec_pts = list(compress(B2_pts,out12))
        else:
            mainB = B2
            sec_pts = list(compress(B1_pts,out21))
        box_out = box_3pts(mainB,sec_pts)
    elif sum(out12)>3:
        box_out = [B1]
    else:
        box_out = [B2]
    return box_out

def sort_list(list1, sort_list):
    zipped_pairs = zip(sort_list, list1)
    z = [x for _, x in sorted(zipped_pairs)] 
    return z

def split_list(boxes):
    i=0
    overlapping = True
    areas=[]
    for bx in boxes:
        areas.append(-1*calc_area(bx))
    sorted_boxes = sort_list(boxes,areas) 
    while overlapping:                   
        large_box = sorted_boxes[i]
        test_boxes = sorted_boxes[i+1:]
        for j, tb in enumerate(test_boxes):
            if overlap(large_box, tb):
                new_boxes = box_split(large_box,tb)
                new_boxes.pop(0)
                sorted_boxes.pop(j+i+1)
                sorted_boxes.extend(new_boxes)
                break
            elif j == len(test_boxes)-1:
                i+=1
                if i == len(sorted_boxes)-1:
                    overlapping = False
        areas =[]
        for bx in sorted_boxes:
            areas.append(-1*calc_area(bx))
        sorted_boxes = sort_list(sorted_boxes,areas)
    return sorted_boxes   
    
    
blk_image = np.zeros((500,500)) 
blk_image1 = blk_image.copy() 
blk_image2 = blk_image.copy() 
blk_image3 = blk_image.copy() 
blk_image4 = blk_image.copy() 
blk_image5 = blk_image.copy()
  
   
d_box = [[200,200,300,300],[250,250,350,350],[230,175,320,340],[180,260,370,290]]
areas = []
for i, bx in enumerate(d_box):    
    areas.append(calc_area(bx))

overlap12 = overlap(d_box[0],d_box[1]) 
out12, B2_pts= outside_box(d_box[0],d_box[1]) 
out13, B3_pts = outside_box(d_box[0],d_box[2])  

box12 = box_2pts(d_box[0],list(compress(B2_pts,out12))) 
box13 = box_3pts(d_box[0],list(compress(B3_pts,out13)))

boxes12 = box_split(d_box[0], d_box[1])
boxes13 = box_split(d_box[2], d_box[0])
boxes23 = box_split(d_box[2], d_box[1])
boxes14 = box_split(d_box[0], d_box[3])
boxes123 = split_list(d_box)

for bbox in d_box:
    start=(bbox[0],bbox[1])
    end = (bbox[2],bbox[3])
    box_test_img = cv2.rectangle(blk_image1, start, end, (255,0,0), 2)

fig, axis = plt.subplots(1, figsize=(12,6))
axis.imshow(box_test_img)

for bbox in boxes123:
    start=(bbox[0],bbox[1])
    end = (bbox[2],bbox[3])
    box_test_img2 = cv2.rectangle(blk_image2, start, end, (255,0,0), 2)

fig, axis = plt.subplots(1, figsize=(12,6))
axis.imshow(box_test_img2)

# for bbox in boxes13:
#     start=(bbox[0],bbox[1])
#     end = (bbox[2],bbox[3])
#     box_test_img3 = cv2.rectangle(blk_image3, start, end, (255,0,0), 2)

# fig, axis = plt.subplots(1, figsize=(12,6))
# axis.imshow(box_test_img3)

# for bbox in boxes23:
#     start=(bbox[0],bbox[1])
#     end = (bbox[2],bbox[3])
#     box_test_img4 = cv2.rectangle(blk_image4, start, end, (255,0,0), 2)

# fig, axis = plt.subplots(1, figsize=(12,6))
# axis.imshow(box_test_img4)

# for bbox in boxes14:
#     start=(bbox[0],bbox[1])
#     end = (bbox[2],bbox[3])
#     box_test_img5 = cv2.rectangle(blk_image5, start, end, (255,0,0), 2)

# fig, axis = plt.subplots(1, figsize=(12,6))
# axis.imshow(box_test_img5)