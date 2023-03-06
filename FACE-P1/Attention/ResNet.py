import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, BatchNormalization, Add, ReLU
from tensorflow.keras.models import Model


def identity_block(x, filters,layer,block):
    """
    Implementation of the identity block.

    Arguments:
    x -- input tensor of shape (batch_size, height, width, channels)
    filters -- tuple of three integers, specifying the number of filters for each Conv2D layer in the block

    Returns:
    output -- output of the identity block, tensor of shape (batch_size, height, width, channels)
    """
    f1, f2, f3 = filters

    x_shortcut = x

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name="conv%s_block%s_1_conv"%(layer,block))(x)
    x = BatchNormalization(name="conv%s_block%s_1_bn"%(layer,block))(x)
    x = ReLU(name="conv%s_block%s_1_relu"%(layer,block))(x)

    x = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name="conv%s_block%s_2_conv"%(layer,block))(x)
    x = BatchNormalization( name="conv%s_block%s_2_bn"%(layer,block))(x)
    x = ReLU(name="conv%s_block%s_2_relu"%(layer,block))(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name="conv%s_block%s_3_conv"%(layer,block))(x)
    x = BatchNormalization( name="conv%s_block%s_3_bn"%(layer,block))(x)

    x = Add()([x, x_shortcut])
    x = ReLU()(x)

    return x


def convolutional_block(x, filters, strides,layer,block):
    """
    Implementation of the convolutional block.

    Arguments:
    x -- input tensor of shape (batch_size, height, width, channels)
    filters -- tuple of three integers, specifying the number of filters for each Conv2D layer in the block
    strides -- tuple of two integers, specifying the strides for the first Conv2D layer in the block

    Returns:
    output -- output of the convolutional block, tensor of shape (batch_size, height, width, channels)
    """
    f1, f2, f3 = filters

    x_shortcut = x

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=strides, padding='valid', name="conv%s_block%s_1_conv"%(layer,block))(x)
    x = BatchNormalization( name="conv%s_block%s_1_bn"%(layer,block))(x)
    x = ReLU(name="conv%s_block%s_1_relu"%(layer,block))(x)
    x = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name="conv%s_block%s_2_conv"%(layer,block))(x)
    x = BatchNormalization( name="conv%s_block%s_2_bn"%(layer,block))(x)
    x = ReLU( name="conv%s_block%s_2_relu"%(layer,block))(x)
    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name="conv%s_block%s_3_conv"%(layer,block))(x)
    x = BatchNormalization(name="conv%s_block%s_3_bn"%(layer,block))(x)
    x_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=strides, padding='valid', name="conv%s_block%s_0_conv"%(layer,block))(x_shortcut)
    x_shortcut = BatchNormalization(name="conv%s_block%s_0_bn"%(layer,block))(x_shortcut)

    x = Add()([x, x_shortcut])
    x = ReLU()(x)

    return x
    
    
    
def layer123(x_input):
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',name="conv1_conv")(x_input)
    x = BatchNormalization(name="conv1_bn")(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    
    x = convolutional_block(x, filters=(64, 64, 256), strides=(1, 1), layer=2, block=1)
    x = identity_block(x, filters=(64, 64, 256), layer=2, block=2)
    x = identity_block(x, filters=(64, 64, 256), layer=2, block=3)
    x1=x
    x = convolutional_block(x, filters=(128, 128, 512), strides=(2, 2), layer=3, block=1)
    x = identity_block(x, filters=(128, 128, 512), layer=3, block=2)
    x = identity_block(x, filters=(128, 128, 512), layer=3, block=3)
    x = identity_block(x, filters=(128, 128, 512), layer=3, block=4)
    return x1, x
    
    
    
def layer45(x):
    
    x = convolutional_block(x, filters=(256, 256, 1024), strides=(2, 2), layer=4, block=1)
    x = identity_block(x, filters=(256, 256, 1024), layer=4, block=2)
    x = identity_block(x, filters=(256, 256, 1024), layer=4, block=3)
    x = identity_block(x, filters=(256, 256, 1024), layer=4, block=4)
    x = identity_block(x, filters=(256, 256, 1024), layer=4, block=5)
    x = identity_block(x, filters=(256, 256, 1024), layer=4, block=6)
    x3=x
    x = convolutional_block(x, filters=(512, 512, 2048), strides=(2, 2), layer=5, block=1)
    x = identity_block(x, filters=(512, 512, 2048), layer=5, block=2)
    x = identity_block(x, filters=(512, 512, 2048), layer=5, block=3)
    
    return x3, x

def ResNet50(input_shape):
    """
    Implementation of the ResNet50 architecture.

    Arguments:
    input_shape -- shape of the input images (height, width, channels)
    num_classes -- number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    x_input = Input(input_shape)
    x1, x2 = layer123(x_input)
    x3,x4 = layer45(x2)
    

    model = Model(inputs=x_input, outputs=[x1,x2,x3,x4], name='ResNet50')
    return model
    
    
def LastLayers(input_shape):
    x_input = Input(input_shape)
    x3,x4 = layer45(x_input)
    model = Model(inputs=x_input, outputs=[x3,x4], name='LastLayers')

    return model