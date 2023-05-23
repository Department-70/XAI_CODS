# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:10:39 2023

@author: Debra Hogue

Description: Decision Hierarchy for CODS XAI
             Lvl 1 - Binary mask evaluation  - Is anything present?
             Lvl 2 - Ranking mask evaluation - Where is the weak camouflage located?
             Lvl 3 - Object Part Identification of weak camouflage - What part of the object breaks the camouflage concealment?
"""

import cv2
import os
import json
import numpy as np
from PIL import Image
import matplotlib.colors as color
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours

""" folder locations for binary maps, ranking maps, and object parts json """
# image_root = './dataset/COD10K_FixTR/Image/'
# gt_root = './dataset/COD10K_FixTR/GT/'
# fix_root = './dataset/COD10K_FixTR/Fix/'
# obj_parts = './dataset/CORVIS-parts-dataset/'

image_root = './dataset/train/Imgs/'
gt_root = './dataset/train/GT/'
fix_root = './dataset/train/Fix/'

# Example image
NonCAM = 'COD10K-NonCAM-5-Background-5-Vegetation-4854.png'
CAM = 'COD10K-CAM-3-Flying-55-Butterfly-3330.png'

# Custom colormap for fixation maps
RdBl = color.LinearSegmentedColormap.from_list('blGrRdBl', ["black", "black", "red", "red"])
# plt.colormaps.register(RdBl)

blGrRdBl = color.LinearSegmentedColormap.from_list('blGrRdBl', ["black", "blue", "green", "red", "red"])
# plt.colormaps.register(blGrRdBl)

if not os.path.exists("figures"):
    os.mkdir("figures")
    
if not os.path.exists("bbox_figures"):
    os.mkdir("bbox_figures")
    
if not os.path.exists("jsons"):
    os.mkdir("jsons")
    
if not os.path.exists("outputs"):
    os.mkdir("outputs")


"""
===================================================================================================
    Helper function
        - Adds the XAI message as a label onto a copy of th esegmented image
===================================================================================================
"""
from PIL import ImageDraw
from PIL import ImageFont

def add_label(image, label_text, label_position):

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define the font and font size
    font = ImageFont.truetype('arial.ttf', 20)

    # Draw the label on the image
    draw.text(label_position, label_text, fill=(255, 255, 255), font=font)

    # Save the image with the label
    image.save('labeled_image.jpg')


"""
===================================================================================================
    Helper function
        - Adds the XAI message as a label onto a copy of the segmented image
===================================================================================================
"""

def segment_image(original_image, mask_image, color=(255, 0, 0)):
    
    # Conver the mask to mode 1
    mask_image = mask_image.convert('1')
    
    # Check that the mask and image have the same size
    if original_image.size != mask_image.size:
        raise ValueError('Image and mask must have the same size.')
        
    # Check that the mask has the correct mode
    if mask_image.mode != '1':
        raise ValueError('Mask must be a binary image.')
        
    # Convert the mask to a numpy array
    mask_array = np.array(mask_image, dtype=bool)
    
    # Create a copy of the original image
    highlighted_image = np.array(original_image).copy()
    
    # Apply the segmentation to the image
    indices = np.nonzero(mask_array)
    highlighted_image[indices] = color
    
    # Convert the numpy array back to a PIL Image
    highlighted_image = Image.fromarray(highlighted_image)

    return highlighted_image


"""
===================================================================================================
    Helper function
        - Reassigns grayscale values to colors: (Hard: Blue, Medium: Green, Weak: Red)
        -- Only the red will be visible, the medium and hard areas will be black
===================================================================================================
"""
def processFixationMap(fix_image):      
    # Input data should range from 0-1
    img_np = np.asarray(fix_image)/255
    
    # Colorize the fixation map
    color_map = Image.fromarray(blGrRdBl(img_np, bytes=True))
    # color_map.show()
    
    return color_map


"""
===================================================================================================
    Helper function
        - Reassigns grayscale values to colors: (Hard: Blue, Medium: Green, Weak: Red)
        -- Only the red will be visible, the medium and hard areas will be black
===================================================================================================
"""
def findAreasOfWeakCamouflage(fix_image):      
    # Input data should range from 0-1
    img_np = np.asarray(fix_image)/255
    
    # Colorize the fixation map
    color_map = Image.fromarray(RdBl(img_np, bytes=True))
    #color_map.show()
    
    return color_map


"""
===================================================================================================
    Helper function
        - Convert a mask to border image
===================================================================================================
"""
def mask_to_border(mask):
    # Convert PIL image (RGB), mask, to cv2 image (BGR), cv_mask
    open_cv_image = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2GRAY)
    
    # Get the height and width    
    h = open_cv_image.shape[0]
    w = open_cv_image.shape[1]
    border = np.zeros((h, w))

    contours = find_contours(open_cv_image, 1)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


"""
===================================================================================================
    Helper function
        - Mask to bounding boxes
===================================================================================================
"""
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes


"""
===================================================================================================
    Helper function
        - Parsing mask for drawing bounding box(es) on an image
===================================================================================================
"""
def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


"""
===================================================================================================
    Lvl 3 - What part of the object breaks the camouflage concealment?
===================================================================================================
"""
def levelThree(fixation_map, message):
    # print("In level 3")
    # print ("TBD")
    # print("")
        
    return message


"""
===================================================================================================
    Lvl 2 - Where is the weak camouflage located?
        Input:  original_image & fixation_map from Lvl 1 (Camouflage Ranking Map)
        Output: original image & list of bounding boxes
===================================================================================================
"""
def levelTwo(filename, original_image, all_fix_map, fixation_map, message):
    
    # Mask the red area(s) in the fixation map (weak camouflaged area(s))
    fig, axis = plt.subplots(1,2, figsize=(12,6))
    axis[0].imshow(original_image);
    axis[0].set_title('Original Image')
    axis[1].imshow(all_fix_map)
    axis[1].set_title('Fixation Map')
    plt.tight_layout()
    
    # Save plot to output folder for paper
    plt.savefig("figures/fig_"+filename)
    
    # Applying bounding box(es) with the original image for output to lvl 3
    bboxes = mask_to_bbox(fixation_map)
    
    # Convert original_image to cv2 image (BGR) and apply the bounding box
    open_cv_orImage1 = original_image.copy()
    open_cv_orImage2 = original_image.copy()
    
    # Crop areas of weak camo into a collection of images
    cropped_images = []
    data = {"item": 
            {
                "name": filename + ".jpg",
                "num_of_weak_areas": len(bboxes)
            },
            "weak_area_bbox": []
        }
    index = 1
    
    # Looping through the bounding boxes
    for bbox in bboxes:
        # Marking red bounding box(es) on the original image
        starting_point = (bbox[0], bbox[1])
        ending_point = (bbox[2], bbox[3])
        marked_image = cv2.rectangle(open_cv_orImage1, starting_point, ending_point, (255,0,0), 2)
        
        # Slicing to crop the image (y1:y2, x1:x2)
        ci = open_cv_orImage2[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        cropped_images.append(ci)
        # cv2.imwrite("bbox_figures/cropped_"+filename+str(index)+".png", ci)
        
        # Create json of bounding box(es)
        data["weak_area_bbox"].append(
            {
                    "x1":bbox[0],
                    "y1":bbox[1],
                    "x2":bbox[2],
                    "y2":bbox[3]
            })
        
        # Increase Index
        index = index + 1
    
    # the json file to save the output data   
    with open("jsons/"+filename+".json", "w")  as f: 
        json.dump(data, f, indent = 6)  
        f.close() 
        
    # Figure of original image and marked image
    fig, axis = plt.subplots(1,2, figsize=(12,6))
    axis[0].imshow(marked_image);
    axis[0].set_title('Identified Weak Camo')
    axis[1].imshow(cropped_images[0])
    axis[1].set_title('Cropped Weak Camo Area')
    
    # Save plot to output folder for paper
    plt.savefig("bbox_figures/fig_"+filename)
    
    message += "Identified " + str(index-1) + " weak camouflaged area(s).  \n"
    
    # Weak camouflaged area annotated image
    output = levelThree(cropped_images, message)
    
    return output


"""
===================================================================================================
    Lvl 1 - Is anything present?
        Input:  binary_map & fix_map
        Output: fix_map (if a camouflaged object is detected)
===================================================================================================
"""
def levelOne(filename, binary_map, all_fix_map, fix_image, original_image, message):

    # Does the numpy array contain any non-zero values?
    all_zeros = not binary_map.any()
    
    if all_zeros:
        # No object detected, no need to continue to lower levels
        message += "No object present. \n"
        # print("No object present.")
        output = message
    else:
        # Object detected, continue to Lvl 2
        message += "Object present. \n"
        # print("Object detected.")
        # print("")
        output = levelTwo(filename, original_image, all_fix_map, fix_image, message)
        
    return output


"""
===================================================================================================
    XAI Function
===================================================================================================
"""
def xaiDecision(file, counter):
    
    # Filename
    file_name = os.path.splitext(file.name)[0]

    # XAI Message
    message = "Decision for " + file_name + ": \n"
    
    # Gather the images: Original, Binary Mapping, Fixation Mapping
    original_image = cv2.imread(image_root + file_name + '.jpg')
    dim = original_image.shape
    
    if os.path.exists(gt_root + file_name + '.png'):
        bm_image = Image.open(gt_root + file_name + '.png')
    else:
        bm_image = np.zeros((dim[1], dim[0],3), np.uint8)
    
    if os.path.exists(fix_root + file_name + '.png'):
        fix_image = Image.open(fix_root + file_name + '.png')
    else:
        fix_image = np.zeros((dim[1], dim[0],3), np.uint8)
    
    # Normalize the Binary Mapping
    trans_img = np.transpose(bm_image)
    img_np = np.asarray(trans_img)/255
    
    # Preprocess the Fixation Mapping
    weak_fix_map = findAreasOfWeakCamouflage(fix_image)
    all_fix_map = processFixationMap(fix_image)
    
    output = levelOne(file_name, img_np, all_fix_map, weak_fix_map, original_image, message)

    org_image = Image.open(image_root + file_name + '.jpg')
    segmented_image = segment_image(org_image, fix_image, color=(255, 0, 0))
    add_label(segmented_image, output, (15, 15))
    segmented_image.save('outputs/segmented_'+ file_name +'.jpg')
    
    return message


"""
===================================================================================================
    Main
===================================================================================================
"""
if __name__ == "__main__":
    # Counter
    counter = 1
    
    # Loop to iterate through dataset
    for files in os.scandir(image_root):
        # Filename
        file_name = os.path.splitext(files.name)[0]
    
        # XAI Message
        message = "Decision for " + file_name + ": \n"
        
        # Gather the images: Original, Binary Mapping, Fixation Mapping
        original_image = cv2.imread(image_root + file_name + '.jpg')
        dim = original_image.shape
        
        if os.path.exists(gt_root + file_name + '.png'):
            bm_image = Image.open(gt_root + file_name + '.png')
        else:
            bm_image = np.zeros((dim[1], dim[0],3), np.uint8)
        
        if os.path.exists(fix_root + file_name + '.png'):
            fix_image = Image.open(fix_root + file_name + '.png')
        else:
            fix_image = np.zeros((dim[1], dim[0],3), np.uint8)
        
        # Normalize the Binary Mapping
        trans_img = np.transpose(bm_image)
        img_np = np.asarray(trans_img)/255
        
        # Preprocess the Fixation Mapping
        weak_fix_map = findAreasOfWeakCamouflage(fix_image)
        all_fix_map = processFixationMap(fix_image)
        
        output = levelOne(file_name, img_np, all_fix_map, weak_fix_map, original_image, message)
    
        org_image = Image.open(image_root + file_name + '.jpg')
        segmented_image = segment_image(org_image, fix_image, color=(255, 0, 0))
        add_label(segmented_image, output, (15, 15))
        segmented_image.save('outputs/segmented_'+ file_name +'.jpg')
        
        counter += 1
        
        if counter == 3041:
            break
    
