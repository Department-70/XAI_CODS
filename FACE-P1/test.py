
import tensorflow as tf
from tensorflow.keras import activations, layers, losses

import numpy as np
import pdb, os, argparse
from scipy import misc
from Attention.ResNet_models import Generator
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()

dataset_path = './dataset/test/'

generator = Generator(channel=opt.feat_channel)
generator.load_weights('./models/Resnet/model')
generator.build((1,480,480,3))
# generator.summary()
generator.compile()

test_datasets = ['Mine']

for dataset in test_datasets:
    save_path = './results/small/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/Imgs/'
    test_loader = test_dataset(image_root, opt.testsize)
    
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