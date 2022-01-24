import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, MaxPool2D, ReLU, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa



class ResNet18(tf.keras.Model):

    def __init__(self, input_shape, bn=None, pool_list=[2,2,2,4]):
        super().__init__()
        self.conv1 = self.conv_block(64, input_shape=input_shape, bn=bn)
        self.res1 = Sequential([self.conv_block(64, bn=bn), self.conv_block(64, bn=bn)])
        self.res2 = Sequential([self.conv_block(64, bn=bn), self.conv_block(64, bn=bn)])
        self.res3 = self.conv_block(128, pool=True, pool_no=pool_list[0], bn=bn)
        self.res4 = self.conv_block(128, bn=bn)
        self.res5 = Sequential([self.conv_block(128, bn=bn), self.conv_block(128, bn=bn)])
        self.res6 = self.conv_block(256, pool=True, pool_no=pool_list[1], bn=bn)
        self.res7 = self.conv_block(256, bn=bn)
        self.res8 = Sequential([self.conv_block(256, bn=bn), self.conv_block(256, bn=bn)])
        self.res9 = self.conv_block(512, pool=True, pool_no=pool_list[2], bn=bn)
        self.res10 = self.conv_block(512, bn=bn)
        self.res11 = Sequential([self.conv_block(512, bn=bn), self.conv_block(512, bn=bn)])
        self.res12 = Sequential([MaxPool2D(pool_size=pool_list[3]), Flatten()])

    def conv_block(self, out_channels, input_shape=None, pool=False, pool_no=2, l2_factor=1e-4, bn=None):
        layers = []
        if input_shape is None:
            layers.append(Conv2D(out_channels, kernel_size=(3, 3), padding='same', use_bias=True, strides=(1, 1), 
                            kernel_initializer=VarianceScaling(),  kernel_regularizer=l2(l2_factor)))
        else:
            layers.append(Conv2D(out_channels, kernel_size=(3, 3), padding='same', use_bias=True, strides=(1, 1), 
                            kernel_initializer=VarianceScaling(),  kernel_regularizer=l2(l2_factor), input_shape=input_shape))
        
        if bn == 'bn':
            layers.append(BatchNormalization(axis=-1))
        elif bn == 'gn':
            layers.append(tfa.layers.GroupNormalization(groups=32, axis=-1))
        elif bn == 'sbn':
            layers.append(BatchNormalization(axis=-1))

        layers.append(ReLU())        
        if pool: layers.append(MaxPool2D(pool_size=(pool_no, pool_no)))

        return Sequential(layers)

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.res1(out) + out
        out = self.res2(out) + out
        
        out = self.res3(out)
        out = self.res4(out) + out
        out = self.res5(out) + out

        out = self.res6(out)
        out = self.res7(out) + out
        out = self.res8(out) + out
        
        out = self.res9(out)
        out = self.res10(out) + out
        out = self.res11(out) + out

        out = self.res12(out)
        return out



class ResNet9(tf.keras.Model):

    def __init__(self, input_shape, bn=None, pool_list=[2,2,2,4]):
        super().__init__()
        self.conv1 = self.conv_block(64, input_shape=input_shape, bn=bn)
        self.res1 = self.conv_block(128, pool=True, pool_no=pool_list[0], bn=bn)
        self.res2 = Sequential([self.conv_block(128, bn=bn), self.conv_block(128, bn=bn)])
        self.res3 = self.conv_block(256, pool=True, pool_no=pool_list[1], bn=bn)
        self.res4 = self.conv_block(512, pool=True, pool_no=pool_list[2], bn=bn)
        self.res5 = Sequential([self.conv_block(512, bn=bn), self.conv_block(512, bn=bn)])
        self.res6 = Sequential([MaxPool2D(pool_size=pool_list[3]), Flatten()])

    def conv_block(self, out_channels, input_shape=None, pool=False, pool_no=2, l2_factor=1e-4, bn=None):
        layers = []
        if input_shape is None:
            layers.append(Conv2D(out_channels, kernel_size=(3, 3), padding='same', use_bias=True, strides=(1, 1), 
                            kernel_initializer=VarianceScaling(),  kernel_regularizer=l2(l2_factor)))
        else:
            layers.append(Conv2D(out_channels, kernel_size=(3, 3), padding='same', use_bias=True, strides=(1, 1), 
                            kernel_initializer=VarianceScaling(),  kernel_regularizer=l2(l2_factor), input_shape=input_shape))
        
        if bn == 'bn':
            layers.append(BatchNormalization(axis=-1))
        elif bn == 'gn':
            layers.append(tfa.layers.GroupNormalization(groups=32, axis=-1))
        elif bn == 'sbn':
            layers.append(BatchNormalization(axis=-1))

        layers.append(ReLU())        
        if pool: layers.append(MaxPool2D(pool_size=(pool_no, pool_no)))

        return Sequential(layers)

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.res1(out)
        out = self.res2(out) + out
        
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out) + out

        out = self.res6(out)
        
        return out

