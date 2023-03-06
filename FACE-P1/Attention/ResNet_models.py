# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:38:43 2023

@author: Debra Hogue

Modified RankNet by Lv et al. to use Tensorflow not Pytorch
and added additional comments to explain methods

Paper: Simultaneously Localize, Segment and Rank the Camouflaged Objects by Lv et al.
"""

import numpy as np
import scipy.stats as st
import tensorflow as tf
from Attention.ResNet import ResNet50, LastLayers
from tensorflow.keras import layers # Parameter, Softmax
import tensorflow.keras.applications.resnet50 as models # instantiates the ResNet50 architecture 

from Attention.HolisticAttention import HA


class Generator(tf.keras.Model):
    def __init__(self, channel):
        super(Generator, self).__init__()
        self.channel = channel
        self.sal_encoder = Saliency_feat_encoder(channel)
        
    def get_config(self):
        return {"channel": self.channel}
    def from_config(self,config):
        return self(**config)

    def call(self, x):
        fix_pred, cod_pred1, cod_pred2 = self.sal_encoder(x)
        shape =tf.slice(tf.shape(x), [1],[2])
        fix_pred = tf.image.resize(fix_pred, size=shape, method=tf.image.ResizeMethod.BILINEAR)
        cod_pred1 = tf.image.resize(cod_pred1, size=shape, method=tf.image.ResizeMethod.BILINEAR)
        cod_pred2 = tf.image.resize(cod_pred2, size=shape, method=tf.image.ResizeMethod.BILINEAR)
        return tf.stack([fix_pred, cod_pred1, cod_pred2])
    

"""
    Position Attention Module (PAM)
    paper: Dual Attention Network for Scene Segmentation
"""
class PAM_Module(layers.Layer):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = layers.Conv2D(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = layers.Conv2D(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = layers.Conv2D(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = layers.Parameter(tf.zeros(1))
        self.softmax = layers.Softmax(dim=-1)

    def get_config(self):
        return {"in_dim": self.channel_in}
    def from_config(self,config):
        return self(**config)

    def call(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = tf.BatchMatMul(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = tf.BatchMatMul(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


"""
    Classifier Module
"""
class Classifier_Module(tf.keras.Model):
    def __init__(self,dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = self.layers
        self.dilation_series=dilation_series
        self.padding_series =padding_series
        self.NoLabels = NoLabels
        self.input_channel=input_channel
        
        for dilation,padding in zip(dilation_series, padding_series):
            l = (layers.ZeroPadding2D(padding),layers.Conv2D(NoLabels, kernel_size=3, strides=1, padding="valid", dilation_rate=dilation, use_bias = True))
        
            self.conv2d_list.append(l)
        for m in self.conv2d_list:
            l = m
            if (isinstance(m, tuple)):
                _, l = m
            initializer = tf.keras.initializers.RandomNormal(0., 0.01)
            l.kernel_initializer =initializer
            
    def get_config(self):
        return {"dilation_series": self.dilation_series, "padding_series" : self.padding_series, "NoLabels" :self.NoLabels, "input_channel":self.input_channel}
    def from_config(self,config):
        return self(**config)
        
    def call(self, x):
        if (isinstance(self.conv2d_list[0],tuple)):
            p,l = self.conv2d_list[0]
            out = l(p(x))
        else:   
            out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            if (isinstance(self.conv2d_list[i+1],tuple)):
                p,l = self.conv2d_list[i+1]
                out += l(p(x))
            else:   
                out += self.conv2d_list[i+1](x)
        return out


"""
    Channel Attention (CA) Layer
"""
class CALayer(layers.Layer):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.channel = channel
        
        self.reduction = reduction
        # global average pooling: feature --> point
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        
        # feature channel downscale and upscale --> channel weight
        self.conv_du = tf.keras.Sequential( layers=[
                layers.Conv2D( channel // reduction, 1, padding="valid", use_bias=True),
                layers.ReLU(),
                layers.Conv2D( channel, 1, padding="valid", use_bias=True),
                layers.Activation("sigmoid")]
        )
    def get_config(self):
        return {"channel": self.channel, "reduction":self.reduction}
    def from_config(self,config):
        return self(**config)
    def call(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


"""
    Residual Channel Attention Block (RCAB)
        input: B*C*H*W
        output: B*C*H*W 
    
    paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks  
"""
class RCAB(layers.Layer):

    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=layers.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        self.config = {"n_feat": n_feat, "kernel_size":kernel_size,"reduction":reduction, "bias": bias, "bn":bn, "act":  act , "res_scale":res_scale}
        
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(layers.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = tf.keras.Sequential(modules_body)
        self.res_scale = res_scale

    def get_config(self):
        return self.config
    def from_config(self,config):
        return self(**config)
        
        
        
    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return layers.Conv2D(out_channels, kernel_size,padding="same", use_bias=bias)

    def call(self, x):
        
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


"""
    Basic 2D Convolution
"""
class BasicConv2d(layers.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.config = {"in_planes": in_planes, "out_planes": out_planes, "kernel_size": kernel_size, "stride":stride, "padding": padding, "dilation":dilation}
        
        self.conv_bn = tf.Sequential(
            layers.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            layers.BatchNorm2d(out_planes)
        )
    def get_config(self):
        return self.config
    def from_config(self,config):
        return self(**config)
    def call(self, x):
        x = self.conv_bn(x)
        return x


"""
    Triple Convolution
"""
class Triple_Conv(layers.Layer):
    def __init__(self, in_channel, out_channel):
        super(Triple_Conv, self).__init__()
        self.config = {"in_channel":in_channel, "out_channel":out_channel}
        self.reduce = tf.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def get_config(self):
        return self.config
    def from_config(self,config):
        return self(**config)
        
    def call(self, x):
        return self.reduce(x)    


"""
   Saliency Features Decoder 
"""
class Saliency_feat_decoder(layers.Layer):
    # Resnet based encoder decoder
    def __init__(self, channel):
        super(Saliency_feat_decoder, self).__init__()
        self.config={"channel":channel}
        self.relu = layers.ReLU()
        self.upsample8 = layers.UpSampling2D(size=(8,8), interpolation='bilinear')
        self.upsample4 = layers.UpSampling2D(size=(4,4), interpolation='bilinear')
        self.upsample2 = layers.UpSampling2D(size=(2,2), interpolation='bilinear')
        self.upsample05 = layers.UpSampling2D(size=(0.5,0.5), interpolation='bilinear')
        self.dropout = layers.Dropout(0.3)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel*4)
        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.conv43 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2*channel)
        self.conv432 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 3*channel)
        self.conv4321 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 4*channel)

        self.cls_layer = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)

    def get_config(self):
        return self.config
    def from_config(self,config):
        return self(**config)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)


    def call(self, x1,x2,x3,x4):

      
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)
        conv4_feat = self.upsample2(conv4_feat)

        conv43 = tf.concat((conv4_feat, conv3_feat),3)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = tf.concat((self.upsample2(conv4_feat), conv43, conv2_feat), 3)
        conv432 = self.racb_432(conv432)
        conv432 = self.conv432(conv432)
        conv432 = self.upsample2(conv432)
        
        #print("1 %s"%(self.upsample4(conv4_feat)))
        #print("2 %s"%(self.upsample2(conv43)))
        #print("3 %s"%(conv432))
        #print("4 %s"%(conv1_feat))
        conv4321 = tf.concat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 3)
        #print("before racb")
        #print(conv4321)
        conv4321 = self.racb_4321(conv4321)
        #print("#################################################################")
        sal_pred = self.cls_layer(conv4321)


        return sal_pred


"""
    Fixation Features Decoder - resnet based encoder decoder
"""
class Fix_feat_decoder(layers.Layer):
    def __init__(self, channel):
        super(Fix_feat_decoder, self).__init__()
        self.config={"channel":channel}
        self.relu = layers.ReLU()
        self.upsample8 = layers.UpSampling2D(size=(8,8), interpolation='bilinear')
        self.upsample4 = layers.UpSampling2D(size=(4,4), interpolation='bilinear')
        self.upsample2 = layers.UpSampling2D(size=(2,2), interpolation='bilinear')
        self.upsample05 = layers.UpSampling2D(size=(0.5,0.5), interpolation='bilinear')
        self.dropout = layers.Dropout(0.3)
        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.racb4 = RCAB(channel * 4)
        
        self.cls_layer = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def get_config(self):
        return self.config
    def from_config(self,config):
        return self(**config)

    def call(self, x1,x2,x3,x4):

        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)
        conv4321 = tf.concat((conv1_feat, self.upsample2(conv2_feat),self.upsample4(conv3_feat), self.upsample8(conv4_feat)),3)
        conv4321 = self.racb4(conv4321)
        #print(conv4321)
        sal_pred = self.cls_layer(conv4321)
        #print(sal_pred)

        return sal_pred


"""
    Saliency Feature Encoder - extracts relevant features from raw images
"""
class Saliency_feat_encoder(layers.Layer):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Saliency_feat_encoder, self).__init__()
        self.config={"channel":channel}
       
        self.relu = layers.ReLU()
        self.upsample8 = layers.UpSampling2D(size=(8,8), interpolation='bilinear')
        self.upsample4 = layers.UpSampling2D(size=(4,4), interpolation='bilinear')
        self.upsample2 = layers.UpSampling2D(size=(2,2), interpolation='bilinear')
        self.upsample05 = layers.AveragePooling2D(pool_size=2)
        self.dropout = layers.Dropout(0.3)
        self.cod_dec = Fix_feat_decoder(channel)
        self.sal_dec = Saliency_feat_decoder(channel)

        self.HA = HA()

        
       
    def get_config(self):
        return self.config
    def from_config(self,config):
        return self(**config)

    def call(self, x ):
        x1,x2,x3,x4 = self.B1_res(x)

        fix_pred = self.cod_dec(x1,x2,x3,x4)
        init_pred = self.sal_dec(x1,x2,x3,x4)
        x2_2 = self.HA(1-tf.math.sigmoid(self.upsample05(fix_pred)), x2)
        x3_2, x4_2 = self.B2_res(x2_2)
        ref_pred = self.sal_dec(x1,x2_2,x3_2,x4_2)

        return self.upsample4(fix_pred),self.upsample4(init_pred),self.upsample4(ref_pred)

    def build(self, shape):
        self.B1_res = ResNet50((shape[1],shape[2],shape[3]))
        self.B2_res = LastLayers((int(shape[1]/8),int(shape[2]/8),512))
        
        self.B1_res.build((shape[1],shape[2],shape[3]))
        self.B2_res.build((int(shape[1]/8),int(shape[2]/8),512))
        
        res50 = models.ResNet50(weights="imagenet")
        resName=[]
        for i in res50.layers:
            resName.append(i.name)
            
        def transWeights(layer):
            k = layer.name
            if k in resName:
                w =  res50.get_layer(k).get_weights()
                layer.set_weights(w)
            
                
        for i in self.B1_res.layers:
            transWeights(i)
        for i in self.B2_res.layers:
            transWeights(i)
            
            
                            
            