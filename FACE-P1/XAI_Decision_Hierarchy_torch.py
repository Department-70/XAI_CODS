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
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from data import test_dataset
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse

from scipy import misc
from model.ResNet_models import Generator
from data_torch import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

#For collecting and writing the json stats file
stats={
       "data": []
       }

#Turning off gpu since loading 2 models takes too much VRAM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

""" folder locations for binary maps, ranking maps, and object parts json """
# image_root = './dataset/COD10K_FixTR/Image/'
# gt_root = './dataset/COD10K_FixTR/GT/'
# fix_root = './dataset/COD10K_FixTR/Fix/'
# obj_parts = './dataset/CORVIS-parts-dataset/'
cods = Generator(channel=32)
cods.load_state_dict(torch.load('./models/Resnet/Model_50_gen.pth'))

cods.cuda()
cods.eval()


PATH_TO_SAVED_MODEL = "models/d7_f/saved_model"

detect_fn = []
with tf.device('/CPU:0'):

    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
#CODS_model = './models/FACE-100/'
#cods = tf.saved_model.load(CODS_model)



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
    draw.text(label_position, label_text, fill='white', font=font, stroke_width=2, stroke_fill='black')

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
    if original_image.size != np.transpose(mask_image).shape:
        print(original_image.size)
        print(mask_image.shape)
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
    print(indices)
    print(highlighted_image.shape)
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
    Helper function
        - Returns overlapping boxes 
        box format [xmin, xmax, ymin, ymax]
===================================================================================================
"""
def overlap(bbox1,bbox2):
    def overlap1D(b1,b2):
        return b1[1] >= b2[0] and b2[1] >= b1[0]
    
    return overlap1D(bbox1[:2],bbox2[:2]) and overlap1D(bbox1[2:],bbox2[2:])





"""
===================================================================================================
    Lvl 3 - What part of the object breaks the camouflage concealment?
===================================================================================================
"""
def levelThree(original_image, bbox, message):

    y_size, x_size, channel = original_image.shape
    
    label_map = ["leg","mouth","shadow","tail","arm","eye"]
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(original_image)
    
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
        
    d_class = []
    d_box = []
    
    for i,s in enumerate(detections['detection_scores'].numpy()[0]):
        if s > 0.3:
            d_class.append(detections['detection_classes'].numpy()[0][i])
            d_box.append(detections['detection_boxes'].numpy()[0][i])
            
    
    fig, axis = plt.subplots(1, figsize=(12,6))
    axis.imshow(original_image);
    axis.set_title('Detected features' + str(len(d_box)))
                
    for i,b in enumerate(detections['detection_boxes'].numpy()[0]):
        if  detections['detection_scores'].numpy()[0][i] > 0.3:
            axis.add_patch(Rectangle((b[1]*x_size, b[0]*y_size),  (b[3]-b[1])*x_size,  (b[2]-b[0])*y_size, label="Test", fill=False, linewidth=2, color=(1,0,0)))
            axis.text(b[1]*x_size, b[0]*y_size-10,label_map[int(detections['detection_classes'].numpy()[0][i])-1] + " " + str(detections['detection_scores'].numpy()[0][i]), fontweight=400, color=(1,0,0))
    weak = []
    for box1 in bbox:
        
        for count, box2 in enumerate(d_box):
            feat = []
            if overlap([box1['x1'], box1['x2'], box1['y1'], box1['y2']], [box2[1]*x_size, box2[3]*x_size, box2[0]*y_size,  box2[2]*y_size]):
                message += "Object's " +str( label_map[int(d_class[count])-1]) + "\n"
                feat.append(str( label_map[int(d_class[count])-1]))
        weak.append(feat)        
        
     
        
    stats["data"].append({"obj": True, "weak":weak})
        
        
        
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
    axis[0].imshow(marked_image)
    axis[0].set_title('Identified Weak Camo')
    if not (cropped_images[0].shape[0] ==0 or cropped_images[0].shape[1]==0):
        axis[1].imshow(cropped_images[0])
    axis[1].set_title('Cropped Weak Camo Area')
    
    # Save plot to output folder for paper
    plt.savefig("bbox_figures/fig_"+filename)
    
    message += "Identified " + str(index-1) + " weak camouflaged area(s).  \n"
    
    # Weak camouflaged area annotated image
    output = levelThree(original_image, data["weak_area_bbox"], message)
    
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
        stats["data"].append({"obj": False, "weak":[]})
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
    plt.show()
    
    return message



def xaiDecision_test(file_path,counter):
    
    save_path = file_path + "Fix" + '/'
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_2 = file_path + "GT" + '/'
    print(save_path_2)
    if not os.path.exists(save_path_2):
        os.makedirs(save_path_2)
    if not os.path.exists(file_path+'results/'):
        os.makedirs(file_path+'results/')

    image_root = file_path
    test_loader = test_dataset(image_root, 480)
    
    for i in range(test_loader.size):
        print(i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        fix_pred,cod_pred1,cod_pred2 = cods.forward(image)

        res = cod_pred2
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        res2 = fix_pred
        res2 = F.upsample(res2, size=[WW,HH], mode='bilinear', align_corners=False)
        res2 = res2.sigmoid().data.cpu().numpy().squeeze()
        res2 = 255*(res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
        '''
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1, 3, 1)
        
        plt.imshow(fix_image)
        plt.axis('off')
        plt.title("First")
        fig.add_subplot(1, 3, 2)
        plt.imshow(res)
        plt.axis('off')
        plt.title("Second")
        fig.add_subplot(1, 3, 3)
        plt.imshow(res2)
        plt.axis('off')
        plt.title("Third")
        plt.show()
        '''
        print(save_path+name)
        cv2.imwrite(save_path_2+name, res)
        cv2.imwrite(save_path+name, res2)
        print()
    for files in os.listdir(file_path):
        print(files)
        if os.path.isfile(os.path.join(file_path, files)):
            # Filename
            file_name = os.path.splitext(files)[0]
            #os.path.splitext(file.name)[0]

            # XAI Message
            message = "Decision for " + file_name + ": \n"
            
            # Gather the images: Original, Binary Mapping, Fixation Mapping
            print(file_path + file_name + '.jpg')
            original_image = cv2.imread(file_path + file_name + '.jpg')
            dim = original_image.shape
            
            
            print("bm")
            if os.path.exists(save_path_2 + file_name + '.png'):
                bm_image = Image.open(save_path_2 + file_name + '.png')
            else:
                bm_image = np.zeros((dim[1], dim[0],3), np.uint8)
            print("fix")
            if os.path.exists(save_path + file_name + '.png'):
                fix_image = Image.open(save_path + file_name + '.png')
            else:
                fix_image = np.zeros((dim[1], dim[0],3), np.uint8)
            
            print("normalize")
            # Normalize the Binary Mapping
            trans_img = np.transpose(bm_image)
            img_np = np.asarray(trans_img)/255
            
            print("preprocess")
            # Preprocess the Fixation Mapping
            weak_fix_map = findAreasOfWeakCamouflage(fix_image)
            all_fix_map = processFixationMap(fix_image)
            '''
            fig.add_subplot(1, 2, 1)
            plt.imshow(weak_fix_map)
            plt.axis('off')
            plt.title("First")
            fig.add_subplot(1, 2, 2)
            plt.imshow(all_fix_map)
            plt.axis('off')
            plt.title("Second")
            plt.show()
            '''
            output = levelOne(file_name, img_np, all_fix_map, weak_fix_map, original_image, message)

            org_image = Image.open(image_root + file_name + '.jpg')
            segmented_image = segment_image(org_image, fix_image, color=(255, 0, 0))
            add_label(segmented_image, output, (15, 15))
            segmented_image.save(file_path+'results/'+ file_name +'.jpg')
            
            with open("stats.json", "w") as outfile:
                outfile.write(stats)
            #return message

"""
===================================================================================================
    Main
===================================================================================================
"""
if __name__ == "__main__":
    # Counter
    counter = 1
    # Loop to iterate through dataset
    test_loader = test_dataset(image_root, 480)
    for i in range(test_loader.size):
        # Filename
        image, HH, WW, name = test_loader.load_data()
        file_name = os.path.splitext(name)[0]
    
        # XAI Message
        message = "Decision for " + file_name + ": \n"
        
        print(file_name)
        original_image = cv2.imread(image_root + file_name + '.jpg')
        
        image = image.cuda()
        fix_pred,cod_pred1,cod_pred2 = cods.forward(image)
        
        fix_image = fix_pred
        fix_image = F.upsample(fix_image, size=[WW,HH], mode='bilinear', align_corners=False)
        fix_image = fix_image.sigmoid().data.cpu().numpy().squeeze()
        fix_image = (fix_image - fix_image.min()) / (fix_image.max() - fix_image.min() + 1e-8)
        

        bm_image = cod_pred1
        bm_image = F.upsample(bm_image, size=[WW,HH], mode='bilinear', align_corners=False)
        bm_image = bm_image.sigmoid().data.cpu().numpy().squeeze()
        bm_image = (bm_image - bm_image.min()) / (bm_image.max() - bm_image.min() + 1e-8)
        

        bm_image2 = cod_pred2
        bm_image2 = F.upsample(bm_image2, size=[WW,HH], mode='bilinear', align_corners=False)
        bm_image2 = bm_image2.sigmoid().data.cpu().numpy().squeeze()
        bm_image2 = (bm_image2 - bm_image2.min()) / (bm_image2.max() - bm_image2.min() + 1e-8)
        
 
        
        # Gather the images: Original, Binary Mapping, Fixation Mapping
        dim = original_image.shape
        
        # Normalize the Binary Mapping 
        trans_img = np.transpose(np.where(bm_image>0.5,1,0))
        img_np = np.asarray(trans_img)
        
        # Preprocess the Fixation Mapping
        weak_fix_map = findAreasOfWeakCamouflage(fix_image)
        all_fix_map = processFixationMap(fix_image)
        
        output = levelOne(file_name, img_np, all_fix_map, weak_fix_map, original_image, message)
    
        org_image = Image.open(image_root + file_name + '.jpg')
        segmented_image = segment_image(org_image, Image.fromarray(fix_image*255), color=(255, 0, 0))
        add_label(segmented_image, output, (15, 15))
        segmented_image.save('outputs/segmented_'+ file_name +'.jpg')
        
        counter += 1
        
        if counter == 3041:
            break

    with open("stats.json", "w") as outfile:
        outfile.write(stats)