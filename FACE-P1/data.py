# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:42:48 2023

@author: Debra Hogue

Modified RankNet by Lv et al. to use Tensorflow not Pytorch
and added additional comments to explain methods

Paper: Simultaneously Localize, Segment and Rank the Camouflaged Objects by Lv et al.
"""

import os
from PIL import Image
import tensorflow.data as data
import tensorflow as tf
import random
import numpy as np
from PIL import ImageEnhance
import math


"""
    Several data augumentation strategies
"""
def cv_random_flip(img, fix, gt):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        fix = fix.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    return img, fix, gt


def randomCrop(image, fix, gt):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), fix.crop(random_region), gt.crop(random_region)


def randomRotation(image, fix, gt):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        fix = fix.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
    return image, fix, gt


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0, sigma=0.15):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomGaussian1(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
            
    return Image.fromarray(img)

"""
    Salient Object Dataset - dataset for training
        The current loader is not using the normalized depth maps for training and test. 
        If you use the normalized depth maps (e.g., 0 represents background and 1 represents foreground.), 
        the performance will be further improved.
"""
class SalObjDataset(tf.keras.utils.Sequence):
    def __init__(self, image_root, gt_root, fix_root, trainsize, batch_size, size_rates):
        self.trainsize = trainsize
        self.batch_size= batch_size
        self.size_rates = size_rates
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.fixs = [fix_root + f for f in os.listdir(fix_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.fixs = sorted(self.fixs)
        self.filter_files()
        self.sub_size = math.ceil(len(self.images)/self.batch_size) 
        self.size = self.sub_size * len(size_rates)
    
    def __getitem__(self, idx):
        # print("Lenghts is %s"%(self.__len__()))
        # print("Getting at index: %s"%(idx))
        batch_x = []
        batch_y = []
        batch_z = []
        index = idx % self.sub_size
        rate_index = int(idx / self.sub_size)
        for i in range(index * self.batch_size, (index + 1) *self.batch_size):
            if (i <len(self.images)):
                x, y, z=self.get_pic(i)
                #item is (x, (y,z)) where:
                    #x is the input image
                    #y is the target output images
                trainsize = int(round(self.trainsize * self.size_rates[rate_index] / 32) * 32)
                if self.size_rates[rate_index] != 1:
                    x = tf.image.resize(x, size=[trainsize, trainsize])
                    y = tf.image.resize(y, size=[trainsize, trainsize])
                    z = tf.image.resize(z, size=[trainsize, trainsize])
                batch_x.append(x)
                batch_y.append(y)
                batch_z.append(z)
        y_res =  tf.stack([tf.convert_to_tensor(batch_y),tf.convert_to_tensor(batch_z)])
        return tf.convert_to_tensor(batch_x), y_res
    
    
    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def get_pic(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        fix = self.binary_loader(self.fixs[index])
        image, fix, gt = cv_random_flip(image, fix, gt)
        image, fix, gt = randomCrop(image, fix, gt)
        image, fix, gt = randomRotation(image, fix, gt)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.resize(image, (self.trainsize, self.trainsize))
        image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')
        image = tf.convert_to_tensor(image)
   
        gt = tf.keras.preprocessing.image.img_to_array(gt)
        gt = tf.image.resize(gt, (self.trainsize, self.trainsize))
        gt /= 256
        gt = tf.convert_to_tensor(gt)
        
        fix = tf.keras.preprocessing.image.img_to_array(fix)
        fix = tf.image.resize(fix, (self.trainsize, self.trainsize))
        fix /= 256
        fix = tf.convert_to_tensor(fix)
        zero = tf.zeros([self.trainsize,self.trainsize,2])
        gt = tf.concat([gt,zero],2)
        fix = tf.concat([fix,zero],2)
        return image, gt, fix

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        assert len(self.images) == len(self.fixs)
        images = []
        gts = []
        fixs = []
        for img_path, gt_path, fix_path in zip(self.images, self.gts, self.fixs):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            fix = Image.open(fix_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                fixs.append(fix_path)
        self.images = images
        self.gts = gts
        self.fixs = fixs

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt, fix):
        assert img.size == gt.size
        assert img.size == fix.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), fix.resize((w, h), Image.NEAREST)
        else:
            return img, gt, fix

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, fix_root, trainsize,batchsize, size_rates):

    dataset = SalObjDataset(image_root, gt_root, fix_root, trainsize,batchsize,size_rates)

   
    return dataset


"""
    Testing Dataset class
"""
class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        
        self.size = len(self.images)
        self.index = 0
        
    def len(self):
        return self.size

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.resize(image, (self.testsize, self.testsize))
        image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, 0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')