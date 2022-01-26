import os
import copy
import math
import time
import random
import numpy as np
import pickle
import socket
import struct
import tensorflow as tf
import matplotlib.pyplot as plt

from sys import getsizeof
from tqdm import tqdm
from datetime import datetime

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Flatten, Reshape 
from tensorflow.keras.layers import UpSampling2D, MaxPool2D, Input, Activation, Conv2D, Dense, Dropout, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, AveragePooling2D

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.applications import imagenet_utils

from tensorflow.keras import backend as K


import tensorflow.keras as tf_keras
import tensorflow.keras.models as tf_models
import tensorflow.keras.layers as tf_layers
import tensorflow.keras.regularizers as tf_regularizers
import tensorflow.keras.initializers as tf_initializers

from nets.resnet import ResNet9, ResNet18, WideResNet28x2

gpus = tf.config.experimental.list_physical_devices('GPU')
num = 0 ################################################################################
tf.config.experimental.set_visible_devices(gpus[num], 'GPU')
tf.config.experimental.set_memory_growth(gpus[num], True)

path = './result'

isExist = os.path.exists(path)
if not isExist:  
    os.makedirs(path)
    
exp = 'fedavg_sl_gn'
seed_num=1001


dataset = 'cifar10' #'svhn'
if dataset == 'cifar10':
    num_test = 300
elif dataset =='svhn':
    num_test = 200

num_label_per_class = 54
batch_size = 32

rounds = 300
local_epochs = 2
num_client = 100
num_active_client=5
num_class = 10
################################################################



if dataset=='cifar10':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
elif dataset == 'svhn':
    train_dataset = tfds.load(name="svhn_cropped", split=tfds.Split.TRAIN)
    test_dataset = tfds.load(name="svhn_cropped", split=tfds.Split.TEST)

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for ex in train_dataset:
        train_images.append(ex['image'])
        train_labels.append(ex['label'])
    for ex in test_dataset:
        test_images.append(ex['image'])
        test_labels.append(ex['label'])

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

np.random.seed(seed_num)
idx = np.random.permutation(len(train_images))
train_images = train_images[idx]
train_labels = train_labels[idx]
np.random.seed(seed_num)
idx = np.random.permutation(len(test_images))
test_images = test_images[idx]
test_labels = test_labels[idx]

images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis = 0)

images = images.astype('float32')
labels = tf.keras.utils.to_categorical(labels, num_class)
images /= 255.0


train_dataset = []
train_labels = []
val_dataset = []
val_labels = []
test_dataset = []
test_labels = []
for i in range(num_class):
    train_dataset.append(images[np.where(labels == 1)[1]==i][:5400, :, :, :])
    train_labels.append(labels[np.where(labels == 1)[1]==i][:5400])
    val_dataset.append(images[np.where(labels == 1)[1]==i][5400:5400+num_test, :, :, :])
    val_labels.append(labels[np.where(labels == 1)[1]==i][5400:5400+num_test])
    test_dataset.append(images[np.where(labels == 1)[1]==i][5400+num_test:, :, :, :])
    test_labels.append(labels[np.where(labels == 1)[1]==i][5400+num_test:])

    
val_images = np.concatenate(val_dataset, axis=0)
val_labels = np.concatenate(val_labels, axis=0)
test_images = np.concatenate(test_dataset, axis=0)
test_labels = np.concatenate(test_labels, axis=0)


train_images_list = []
train_labels_list = []

for i in range(num_client):
    client_images = []
    client_labels = []
    
    for c in range(num_class):
        client_images.append(train_dataset[c][i*num_label_per_class : (i+1) * num_label_per_class, :, :, :])
        client_labels.append(train_labels[c][i*num_label_per_class : (i+1) * num_label_per_class])
    
    client_images = np.concatenate(client_images, axis=0)
    client_labels = np.concatenate(client_labels, axis=0)
    
    idx = np.random.permutation(len(client_images))
    client_images = client_images[idx]
    client_labels = client_labels[idx]

    train_images_list.append(client_images)
    train_labels_list.append(client_labels)
    
print("--------------------One client has total {} images\n".format(train_images_list[0].shape[0]))
    
    



'''
def get_model():
    def conv_block(out_channels, pool=False, pool_no=2):
        layers = [tf_layers.Conv2D(out_channels, kernel_size=(3, 3), padding='same', use_bias=True, strides=(1, 1),
                                        kernel_initializer=tf_initializers.VarianceScaling(),  kernel_regularizer=tf_regularizers.l2(1e-4)),
                        BatchNormalization(axis=-1),
                        tf.keras.layers.ReLU()]
        if pool: layers.append(tf_layers.MaxPooling2D(pool_size=(pool_no, pool_no)))
        return tf_models.Sequential(layers)
    inputs = tf_keras.Input(shape=(32,32,3))
    out = conv_block(64)(inputs)
    out = conv_block(128, pool=True, pool_no=2)(out)
    out = tf_models.Sequential([conv_block(128), conv_block(128)])(out) + out

    out = conv_block(256, pool=True)(out)
    out = conv_block(512, pool=True, pool_no=2)(out)
    out = tf_models.Sequential([conv_block(512), conv_block(512)])(out) + out

    out = tf_models.Sequential([tf_layers.MaxPooling2D(pool_size=4),tf_layers.Flatten(), tf_layers.Dense(num_class, use_bias=True, activation='softmax')])(out)
    model = tf_keras.Model(inputs=inputs, outputs=out)    
    return model
'''
# get model
def get_model(model_name='res9', input_shape=(32,32,3)):
    
    #Define downsample sizes
    if input_shape[0] == 32:
        pool_list = [2,2,2,4]
    elif input_shape[0] == 96:
        pool_list=[3,2,4,4]
    
    bn_type = 'gn'
    if model_name == 'res9':    
        model = ResNet9(input_shape=input_shape, bn=bn_type, pool_list=pool_list)
    elif model_name == 'res18':
        model = ResNet18(input_shape=input_shape, bn=bn_type, pool_list=pool_list)
    elif model_name == 'wres28x2':
        model = WideResNet28x2(input_shape=input_shape, bn=bn_type, pool_list=pool_list)

    dummy_in = tf.convert_to_tensor(np.random.random((1,) + input_shape))
    out = model(dummy_in) 
    
    return model


def build_global_model():
    
    return get_model()
    

class Server:
    def __init__(self, 
                 global_model, 
                 optimizer, 
                 loss_fn,
                 val_images,
                 val_labels,
                 test_images,
                 test_labels):
        self.global_model = global_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.val_images = val_images
        self.val_labels = val_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.batch_size = batch_size
        
        self.global_model.compile(optimizer=self.optimizer,
              loss=self.loss_fn,
              metrics=['accuracy'])
        self.val_acc_list = []
        self.client_model_weights_list = []
    
    
    # return: global model weights
    def get_global_model_weights(self):
        return self.global_model.get_weights()
    
    
    # input: client model weight
    # save client model weights to list
    def rec_client_model_weights(self, client_model_weights):
        self.client_model_weights_list.append(client_model_weights)
        
    def fed_avg(self):
        # FedAvg client weights   
        global_weights = list()
        for weights_list_tuple in zip(*self.client_model_weights_list): 
            global_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        self.global_model.set_weights(global_weights)
        
        self.client_model_weights_list = []

        
    # FedAvg and evaluate global model with validation set
    def val_accuracy(self, r):       
        #val_acc = self.global_model.evaluate(x=self.val_images, y=self.val_labels, verbose=1)
        #self.val_acc_list.append(val_acc)
        
        pred = self.global_model.predict(self.val_images)
        pred_label = np.argmax(pred, axis=1)
        label = np.argmax(self.val_labels, axis=1)
        acc = np.sum(pred_label == label) / len(label)
        loss = self.loss_fn(self.val_labels, pred).numpy()
        
        val_acc = [loss, acc]
        self.val_acc_list.append(val_acc)     
        
        print("-----Val Acc: {}, Val Loss: {}".format(acc, loss))

        with open(path + '/' +exp+'_val_acc' , 'a+') as f:
            f.write("{},{},{}\n".format(r, acc, loss))
        return acc
        
    
    # evalutate global model with test sets
    # return: result of evaluate
    def test_accuracy(self, r):
        #hist = self.global_model.evaluate(x=self.test_images, y=self.test_labels, verbose=1)
        
        pred = self.global_model.predict(self.test_images)
        pred_label = np.argmax(pred, axis=1)
        label = np.argmax(self.test_labels, axis=1)
        acc = np.sum(pred_label == label) / len(label)
        loss = self.loss_fn(self.test_labels, pred).numpy()
        
        test_acc = [loss, acc]
        print("-----Test Acc: {}, Test Loss: {}".format(acc, loss))

        with open(path + '/' +exp+'_test_acc' , 'a+') as f:
            f.write("{},{},{}\n".format(r, acc, loss))
                
        return test_acc
        

        
class Client:
    def __init__(self, optimizer, loss_fn):

        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    # input: global model weights
    # training using global model weights
    # return: update client model weights
    def training(self, client_images, client_labels, client_model, global_model_weights, local_epoch, batch_size):
        '''
        client_model.compile(optimizer=self.optimizer,
              loss=self.loss_fn,
              metrics=['accuracy'])
        '''
        client_model.set_weights(global_model_weights)
        
        #K.set_value(self.client_model.optimizer.learning_rate, lr)
        data_num = client_images.shape[0]
        train_step = data_num//batch_size
        
        for e in range(local_epoch):
            for s in range(train_step):
                with tf.GradientTape() as tape:
                    predictions = client_model(client_images[s*batch_size:(s+1)*batch_size], training=True)
                    loss = self.loss_fn(client_labels[s*batch_size:(s+1)*batch_size], predictions)
                grads = tape.gradient(loss, client_model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, client_model.trainable_weights))
            if train_step * batch_size < data_num:
                with tf.GradientTape() as tape:
                    predictions = client_model(client_images[train_step*batch_size:], training=True)
                    loss = self.loss_fn(client_labels[train_step*batch_size:], predictions)
                grads = tape.gradient(loss, client_model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, client_model.trainable_weights))

        del global_model_weights

        return client_model.get_weights()#, hist



    
##########################################################################################################
server = Server(build_global_model(),
                RMSprop(lr=1e-3),
                tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                val_images,
                val_labels,
                test_images,
                test_labels)


client_list = []

for c in range(num_client):
    client_list.append(
        Client(RMSprop(lr=1e-3),
               tf.keras.losses.CategoricalCrossentropy(from_logits=False))
              )

#server.global_model = tf.keras.models.load_model('./save/iidFL/iid_FL_mv2_359')

    
    
print("Training Start")
max_val = 0
cycle = 10
client_model = build_global_model()
for r in range(0, rounds):
    round_start = time.time()
    print("Round {}".format(r))    
    
    # get global model weights
    global_model_weights = server.get_global_model_weights()
    
    random.seed(r+seed_num)
    client_idx = random.sample(range(num_client), num_active_client)
    
    # for each client
    for c in range(num_active_client):

        # training with global model 
        # then send updated client model weights to server
        w = client_list[client_idx[c]].training(
            train_images_list[client_idx[c]],
            train_labels_list[client_idx[c]],
            client_model,
            copy.deepcopy(global_model_weights),
            local_epochs,
            batch_size)
        server.rec_client_model_weights(w)
    
    # FedAvg 
    server.fed_avg()
    # then evaluate with validation set(test set)
    server.val_accuracy(r+1)
    server.test_accuracy(r+1)
  
    round_end = time.time()
    
    print("Time for Round {}: {}".format(r, round_end-round_start))
    
    

