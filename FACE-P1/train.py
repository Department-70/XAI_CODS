# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:11:12 2023

@author: Debra Hogue

Modified RankNet by Lv et al. to use Tensorflow not Pytorch
and added additional comments to explain methods

Paper: Simultaneously Localize, Segment and Rank the Camouflaged Objects by Lv et al.
"""

import tensorflow as tf
from tensorflow.keras import activations, layers, losses
import numpy as np
import os, argparse
from datetime import datetime
from Attention.ResNet_models import Generator
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
import cv2
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import tensorflow.keras.applications.resnet50 as models # instantiates the ResNet50 architecture 

from utils import l2_regularisation

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=480, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))

# build models
generator = Generator(channel=opt.feat_channel)


# generator_params = generator.parameters()
# generator_optimizer = tf.optimizers.Adam(generator_params, opt.lr_gen)


image_root = './dataset/train/Imgs/'
gt_root = './dataset/train/GT/'
fix_root = './dataset/train/Fix/'

# train_loader = get_loader(image_root, gt_root, fix_root,batchsize=opt.batchsize, trainsize=opt.trainsize)
# total_step = len(train_loader)

CE = losses.BinaryCrossentropy(from_logits=True)
mse_loss = losses.MeanSquaredError()
size_rates = [0.75,1,1.25]  # multi-scale training

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

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_cod1(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_cod1.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_cod2(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_cod2.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_fix(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_fix.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_fix_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_fix_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)
        
             
def loss_function(y_true,y_pred):
    gts, fixs = tf.unstack(y_true,2,0)
    gts, _ = tf.split(gts, [1,2], 3)
    fixs, _ = tf.split(fixs, [1,2], 3)
    fix_pred, cod_pred1, cod_pred2 = tf.unstack(y_pred,num=3,axis=0)
    fix_loss = mse_loss(tf.keras.activations.sigmoid(fix_pred),fixs)
    cod_loss1 = structure_loss(cod_pred1, gts)
    cod_loss2 = structure_loss(cod_pred2, gts)
    test= fix_loss + cod_loss1 + cod_loss2
    print("test")
    print(fix_loss)
    print(cod_loss1)
    print(cod_loss2)
    return  fix_loss + cod_loss1 + cod_loss2
    
    
def on_epoch_end( epoch, lr):
    decay = decay_rate ** (epoch // decay_epoch)
    new_lr = lr * decay
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, lr, new_lr))
    return new_lr
          
        
if __name__ == '__main__':
    
  
    
    
    
    
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    op= tf.keras.optimizers.Adam(learning_rate=opt.lr_gen, name='Adam')
    
    generator.compile(optimizer=op, loss=loss_function);
    
    data = get_loader(image_root, gt_root, fix_root, opt.trainsize,opt.batchsize, size_rates)
    
  
    generator.fit(x=data,batch_size=opt.batchsize, epochs=opt.epoch, verbose='auto' , callbacks=[tensorboard_callback,  tf.keras.callbacks.LearningRateScheduler(on_epoch_end)])
     
    save_path = 'models/Resnet/'
    


    #if not os.path.exists(save_path):
     #   os.makedirs(save_path)
    
    #generator.save_weights(save_path+"model")


    dataset_path = './dataset/test/'



    test_datasets = ['Mine']

    for dataset in test_datasets:
        save_path = './results/ResNet50/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    image_root = dataset_path + dataset + '/Imgs/'
    test_loader = test_dataset(image_root, 480)

    for i in range(test_loader.size):
    print(i)
    image, HH, WW, name = test_loader.load_data()
    ans = generator(image)
    _,generator_pred, _  = tf.unstack(ans,num=3,axis=0)
    res = generator_pred
    res = tf.image.resize(res, size=tf.constant([WW,HH]), method=tf.image.ResizeMethod.BILINEAR)
    res = tf.math.sigmoid(res).numpy().squeeze()
    res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
    print(save_path+name)
    cv2.imwrite(save_path+name, res)
    print()