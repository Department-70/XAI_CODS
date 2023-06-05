# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 08:43:15 2023

@author: Debra Hogue

Find and Acquire Camouflage Explainability (FACE) - Phase 1
"""

import tensorflow as tf
import numpy as np
import os, argparse
import cv2
import XAI_Decision_Hierarchy_onxx as xai
#import XAI_Decision_Hierarchy as xai
from data import test_dataset
import glob
from Attention.ResNet_models import Generator
from tensorflow.keras import losses


"""
===================================================================================================
    Parser Creation - for use with scripts to train/test
===================================================================================================
"""

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=480, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--test',type=str,default="./testing_data/default/",help="location of testing data")
opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))

# Loss
CE = losses.BinaryCrossentropy(from_logits=True)
mse_loss = losses.MeanSquaredError()
size_rates = [0.75,1,1.25]  # multi-scale training


# Build models
generator = Generator(channel=opt.feat_channel)


"""
===================================================================================================
    Input
        COD10K_Rank_TR - Training set from COD10K containing 6000 images (CAM & NonCAM) with Rank Map
        
===================================================================================================
"""

""" folder locations for binary maps, ranking maps, and object parts json """
image_root = './dataset/train/Imgs/'
gt_root = './dataset/train/GT/'
fix_root = './dataset/train/Fix/'
# obj_parts = './dataset/CORVIS-parts-dataset/'


"""
===================================================================================================
    Helper Function - structure_loss
        Input:  Prediction & Mask
        Output: Loss Value
===================================================================================================
"""
def structure_loss(pred, mask):
    padded = tf.pad(mask,tf.constant([[0,0],[15,15],[15,15],[0,0]]))
    pooled =tf.nn.avg_pool2d(padded, ksize=31, strides=1, padding="VALID")
    weit  = 1+5*tf.abs(pooled-mask)
    weit = tf.squeeze(weit)
    wbce= tf.nn.sigmoid_cross_entropy_with_logits(mask,pred)
   
    wbce= tf.math.reduce_mean(wbce)
    wbce  = tf.math.reduce_sum((weit*wbce),axis=[1,2]) /tf.reduce_sum(weit,axis=[1,2])
    mask =tf.squeeze(mask)
    pred  = tf.math.sigmoid(pred)
    pred = tf.squeeze(pred)
    inter = tf.math.reduce_sum((pred*mask)*weit,axis=[1,2])
    union = tf.math.reduce_sum((pred+mask)*weit, axis=[1,2])
    wiou  = 1-(inter+1)/(union-inter+1)
    return tf.math.reduce_mean(wbce+wiou)

"""
===================================================================================================
    Helper Function - Visualize the Ground Truth
        This code defines a function `visualize_gt` that takes as input a variable map (`var_map`) 
        and visualizes it by iterating over its first dimension. For each element of this dimension, 
        the function extracts the corresponding 3D tensor (`pred_edge_kk`) and converts it to a 
        numpy array (`pred_edge_kk.detach().cpu().numpy().squeeze()`). It then rescales the pixel 
        values by multiplying them with 255 and converts them to 8-bit integers 
        (`pred_edge_kk *= 255.0` and `pred_edge_kk.astype(np.uint8)`). Finally, it saves the 
        resulting image as a PNG file with a name based on the iteration index (`kk`) in a folder 
        called `temp`.
===================================================================================================
"""
def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


"""
===================================================================================================
    Helper Function - Visualize the COD1
        This function is used to visualize the data in "var_map" by saving the slices of the 
        tensor as images to a directory named "temp" with filenames in the format of "xx_cod1.png" 
        where "xx" is the slice number..
===================================================================================================
"""
def visualize_cod1(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_cod1.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


"""
===================================================================================================
    Helper Function - Visualize the COD2
        This function takes a tensor as input and saves each slice of the tensor as an 8-bit 
        grayscale image in the './temp/' directory with file names that include the slice number 
        and the suffix '_cod2.png'.
===================================================================================================
"""
def visualize_cod2(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_cod2.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


"""
===================================================================================================
    Helper Function - Visualize Fixation Map
        This function takes a tensor as input and saves each slice of the tensor as an 8-bit 
        grayscale image in the './temp/' directory with file names that include the slice number 
        and the suffix '_fix.png'.
===================================================================================================
"""
def visualize_fix(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_fix.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


"""
===================================================================================================
    Helper Function - Visualize Ground Truth Fixation Map
        This function takes a tensor as input and saves each slice of the tensor as an 8-bit 
        grayscale image in the './temp/' directory with file names that include the slice number 
        and the suffix '_fix_gt.png'.
===================================================================================================
"""
def visualize_fix_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_fix_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)
        

"""
===================================================================================================
    Helper Function - loss_function
        Input:  Ground Truth & Prediction
        Output: Loss Value
===================================================================================================
"""
def loss_function(y_true,y_pred):
    
    gts, fixs = tf.unstack(y_true,2,0)
    gts, _ = tf.split(gts, [1,2], 3)
    fixs, _ = tf.split(fixs, [1,2], 3)
    fix_pred, cod_pred1, cod_pred2 = tf.unstack(y_pred,num=3,axis=0)
    fix_loss = mse_loss(tf.keras.activations.sigmoid(fix_pred),fixs)
    cod_loss1 = structure_loss(cod_pred1, gts)
    cod_loss2 = structure_loss(cod_pred2, gts)
    test = fix_loss + cod_loss1 + cod_loss2
    
    # For troubleshooting
    print(test)
    print(fix_loss)
    print(cod_loss1)
    print(cod_loss2)
    
    return  fix_loss + cod_loss1 + cod_loss2


"""
===================================================================================================
    Helper Function - on_epoch_end
        Input:  Epoch & Learning Rate
        Output: New Learning Rate
===================================================================================================
"""
def on_epoch_end(epoch, lr):
    decay = 0.9 ** (epoch // 40) # decay_rate ** (epoch // decay_epoch)
    new_lr = lr * decay
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, lr, new_lr))
    return new_lr


"""
===================================================================================================
    Main
===================================================================================================
"""

from data import get_loader
from datetime import datetime


if __name__ == "__main__":
      
    # Counter
    counter = 1
    
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    """
    =============================
        Training Model
    =============================
    """
        
    # for file in os.listdir('/models/Resnet/'):
    #     if not file.endswith(".pth"):
    
    #         op= tf.keras.optimizers.Adam(learning_rate=opt.lr_gen, name='Adam')
            
    #         generator.compile(optimizer=op, loss=loss_function);
            
    #         data = get_loader(image_root, gt_root, fix_root, opt.trainsize, opt.batchsize, size_rates)
            
          
    #         generator.fit(x=data,
    #                       batch_size=opt.batchsize, 
    #                       epochs=opt.epoch, 
    #                       verbose='auto', 
    #                       callbacks=[tensorboard_callback,  
    #                                  tf.keras.callbacks.LearningRateScheduler(on_epoch_end)])
             
    #         save_path = 'models/Resnet/'


    """
    =============================
        Testing Model
    =============================
    """
    '''
    #Wrote that in to path checking and model checking not really needed
    test_datasets = ['Mine', 'CAMO']
    dataset_path = '../dataset/test/'
    CODS_model = './models/FACE-100/'
    generator2 =  tf.keras.models.load_model(CODS_model, custom_objects={'loss_function': loss_function})
    for dataset in test_datasets:
        save_path = './results/small/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image_root = dataset_path + dataset + '/Imgs/'
        test_loader = test_dataset(image_root, 480)

        for i in range(test_loader.size):
            print(i)
            image, HH, WW, name = test_loader.load_data()
            ans = generator2(image)
            _,generator_pred, _  = tf.unstack(ans,num=3,axis=0)
            res = generator_pred
            res = tf.image.resize(res, size=tf.constant([WW,HH]), method=tf.image.ResizeMethod.BILINEAR)
            res = tf.math.sigmoid(res).numpy().squeeze()
            res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
            print(save_path+name)
            cv2.imwrite(save_path+name, res)
            print()
    '''
        # Loop to iterate through dataset
    testing_dir = opt.test
    #root, dirs, files = os.walk(testing_dir)
    #for f in files:
    #    message = xai.xaiDecision_test(testing_dir,f, counter)   
    #    counter += 1
    
  
    message = xai.xaiDecision_test(testing_dir, counter)   
    counter += 1
        # skip directories
        
        #if counter == 101:#3041:
        #    break
    
    """
    ===================================================================================================
        Decision Generator/Reasoning/Explanation Generator
            Input: Original Image, Camouflage Instance Segmentation & Camouflage Ranking Map
            Output: Segmentation with Explanation
    ===================================================================================================
    """
    
    # Loop to iterate through dataset
    for files in os.scandir(image_root):
        message = xai.xaiDecision(files, counter)   
        counter += 1
        
        if counter == 101:#3041:
            break