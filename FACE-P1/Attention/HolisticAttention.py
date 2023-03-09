# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:18:14 2023

@author: Debra Hogue

Modified RankNet by Lv et al. to use Tensorflow not Pytorch
and added additional comments to explain methods

Paper: Simultaneously Localize, Segment and Rank the Camouflaged Objects by Lv et al.
"""

import numpy as np
import scipy.stats as st
import tensorflow as tf

from tensorflow.keras import layers


""" 
   gkern = Gaussian Kernel matrix 
   kernlen = kernel side length
   nsig = a sigma
   
   A Gaussian Kernel is used in image processing for computing the weighted average 
   of the neighboring points (pixels) in an image
"""
def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


"""
    MinMax Normalization is used to change the range of pixel intensity values
    It is useful here to increase the contrast of the concealed object(s)
"""
def min_max_norm(in_):
    max_ = tf.broadcast_to(tf.math.reduce_max(tf.math.reduce_max(in_,3,True),2,True),tf.shape(in_))
    min_ = tf.broadcast_to(tf.math.reduce_min(tf.math.reduce_min(in_,3,True),2,True),tf.shape(in_))
    in_ = in_ - min_
    return tf.math.truediv(in_, (max_-min_+1e-8))


"""
    Holistic Attention Module
"""
class HA(layers.Layer):
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[  ...,np.newaxis,np.newaxis]
        self.gaussian_kernel = tf.convert_to_tensor(gaussian_kernel)

    def call(self, attention, x):
        attention = tf.cast(attention, tf.float32)
        padded = tf.pad(attention,tf.constant([[0,0],[15,15],[15,15],[0,0]]))

        x = tf.cast(x,tf.float32)
        soft_attention = tf.nn.conv2d(input=padded, filters=self.gaussian_kernel,strides=1,padding='VALID', data_format='NHWC')
        soft_attention = min_max_norm(soft_attention)
        x = tf.math.multiply(x, tf.math.maximum(soft_attention, attention))
        
        return x