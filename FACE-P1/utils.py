# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:35:26 2023

@author: Debra Hogue

Modified RankNet by Lv et al. to use Tensorflow not Pytorch
and added additional comments to explain methods

Paper: Simultaneously Localize, Segment and Rank the Camouflaged Objects by Lv et al.
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras import layers


"""
    Clip gradient
"""
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


"""
    Adjust loss rate
"""
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


"""
    Truncated normal
"""
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


"""
    Initialize weights
"""
def init_weights(m):
    if type(m) == layers.Conv2d or type(m) == layers.ConvTranspose2D:
        tf.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)


"""
    Initialize Orthogonal Normal Weights
"""
def init_weights_orthogonal_normal(m):
    if type(m) == layers.Conv2D or type(m) == layers.ConvTranspose2D:
        layers.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)


"""
    L2 Regularization
"""
def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


"""
    Average meter used to keep a record of loss
"""
class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        a = len(self.losses)
        b = np.maximum(a-self.num, 0)
        c = self.losses[b:]
        #print(c)
        #d = tf.mean(tf.stack(c))
        #print(d)
        return tf.mean(tf.stack(c))

# def save_mask_prediction_example(mask, pred, iter):
# 	plt.imshow(pred[0,:,:],cmap='Greys')
# 	plt.savefig('images/'+str(iter)+"_prediction.png")
# 	plt.imshow(mask[0,:,:],cmap='Greys')
#     plt.savefig('images/'+str(iter)+"_mask.png")
