from imgaug import augmenters as iaa
import os
import sys
import copy
import math
import time
import random
import numpy as np
import pickle
import socket
import struct

from sys import getsizeof
from tqdm import tqdm
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
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


gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) != 0:
    print('use gpu')
    num = 0
    tf.config.experimental.set_visible_devices(gpus[num], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[num], True)
else:
    print('not use gpu')


exp_name = sys.argv[1]
fl_framework = sys.argv[2]
iid = sys.argv[3]
    
path = './result'
exp = exp_name

isExist = os.path.exists(path)
if not isExist:  
    os.makedirs(path)
    
seed_num=1001
random.seed(seed_num)
rounds = 300 # number of rounds
num_client = 100 # number of clients
num_active_client=5 # number of active clients
num_class = 10 # number of class of data
if iid == 'iid':
    is_iid = True
else:
    is_iid = False
num_label = 5 # number of labeled data per each class
num_unlabel = 540 - num_label * 10 # number of unlabeled data
mu_prox = 1e-1 # weight for FedProx 
lr = 1e-3 #learning rate
randaug_n = 2 # number of transformation 
randaug_m = 14 # transformation range
translate_range = (-0.125, 0.125)
weight_unlabel = 1e-2 #weight for unlabeled data
threshold = 0.95 # threshold to determine wheter to use unlabeled data
batch_size = 10 # batch size of labeled data
mu_unlabel = 10 # batch size of unlabeld data
local_epochs = 2 # local epoch


def get_dataset():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    np.random.seed(seed_num)
    idx = np.random.permutation(len(train_images))
    train_images = train_images[idx]
    train_labels = train_labels[idx]
    np.random.seed(seed_num)
    idx = np.random.permutation(len(test_images))
    test_images = test_images[idx]
    test_labels = test_labels[idx]

    images = np.concatenate((train_images, test_images), axis = 0)
    labels = np.concatenate((train_labels, test_labels), axis = 0)

    images = images.astype('float32')
    labels = tf.keras.utils.to_categorical(labels, num_class)

    images /= 255.0

    train_dataset = []
    val_dataset = []
    val_labels = []
    test_dataset = []
    test_labels = []
    for i in range(10):
        train_dataset.append(images[np.where(labels == 1)[1]==i][:5400, :, :, :])
        val_dataset.append(images[np.where(labels == 1)[1]==i][5400:5700, :, :, :])
        val_labels.append(labels[np.where(labels == 1)[1]==i][5400:5700])
        test_dataset.append(images[np.where(labels == 1)[1]==i][5700:, :, :, :])
        test_labels.append(labels[np.where(labels == 1)[1]==i][5700:])
    val_images = np.concatenate(val_dataset, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    test_images = np.concatenate(test_dataset, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    return train_dataset, val_images, val_labels, test_images, test_labels

def get_data_distribution(is_iid=True):
    if is_iid:
        ratio = [[0.5 for _ in range(10)] for _ in range(10)]


        for i in ratio:
            for j in range(10):
                if i[j] == 0.5:
                    i[j] = num_unlabel//10
    
        return ratio
    else:
        ratio = [
            [0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15], # type 0
            [0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03], # type 1 
            [0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03], # type 2 
            [0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03], # type 3 
            [0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02], # type 4 
            [0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03], # type 5 
            [0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03], # type 6 
            [0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03], # type 7 
            [0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15], # type 8 
            [0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50], # type 9
        ]

        for i in ratio:
            for j in range(10):
                if i[j] == 0.5:
                    i[j] = 244
                elif i[j] == 0.15:
                    i[j] = 73
                elif i[j] == 0.03:
                    i[j] = 15
                elif i[j] == 0.02:
                    i[j] = 10
    
        return ratio

def get_client_dataset(dataset, ratio):            
    client_dataset = []

    for _ in range(num_client):
        temp = {}
        for i in range(10):
            temp[str(i)] = []
        temp['unlabel'] = []
        client_dataset.append(temp)

    for label in range(10):
        num_data = len(dataset[label])
        idx = 0
        # distribute label
        for client in range(100):
            for _ in range(num_label):
                client_dataset[client][str(label)].append(dataset[label][idx])
                idx += 1
                
        # distribute unlabel
        for client in range(100):        
            if client != 99:
                for _ in range(ratio[client%10][label]):
                    client_dataset[client]['unlabel'].append(dataset[label][idx])
                    idx += 1
            else:
                num_remain = num_data-idx
                for _ in range(num_remain):
                    client_dataset[client]['unlabel'].append(dataset[label][idx])
                    idx += 1
    
    client_images_list = []
    client_labels_list = []
    client_unlabels_list = []
    
    for i in range(num_client):
        temp_image = []
        temp_label = []
        for key in client_dataset[i]:
            if key == 'unlabel':
                continue
            temp_image.extend(client_dataset[i][key])
            temp_label.extend([int(key) for _ in range(len(client_dataset[i][key]))])
            if i == (num_client-1):
                print("# of labeled data (class {}): {}".format(key, len(client_dataset[i][key])))
        temp_image = np.array(temp_image)
        temp_label = np.array(temp_label)
        temp_label = tf.keras.utils.to_categorical(temp_label, num_class)
        np.random.seed(seed_num)
        idx = np.random.permutation(len(temp_image))
        temp_image = temp_image[idx]
        temp_label = temp_label[idx]
        client_images_list.append(temp_image)
        client_labels_list.append(temp_label)
        random.shuffle(client_dataset[i]['unlabel'])
        client_dataset[i]['unlabel'] = np.array(client_dataset[i]['unlabel'])
        client_unlabels_list.append(client_dataset[i]['unlabel'])
        
        if i == num_client-1:
            print("# of unlabeled data: {}".format(len(client_dataset[i]['unlabel'])))

    
    return client_images_list, client_labels_list, client_unlabels_list
    
    
train_dataset, val_images, val_labels, test_images, test_labels = get_dataset()
data_distribution = get_data_distribution(is_iid)
train_images_list, train_labels_list, train_unlabels_list = get_client_dataset(dataset=train_dataset, ratio=data_distribution)


def get_model(model_arch='resnet8'):
    def conv_block(out_channels, pool=False, pool_no=2):
        layers = [tf_layers.Conv2D(out_channels, kernel_size=(3, 3), padding='same', use_bias=True, strides=(1, 1),
                                        kernel_initializer=tf_initializers.VarianceScaling(),  kernel_regularizer=tf_regularizers.l2(1e-4)),
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



def build_global_model():
    
    return get_model()
    

def difference_model_norm_2_square(global_model, local_model):
    """Calculates the squared l2 norm of a model difference (i.e.
    local_model - global_model)
    Args:
        global_model: the model broadcast by the server
        local_model: the current, in-training model

    Returns: the squared norm

    """
    model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                           local_model,
                                           global_model)
    squared_norm = tf.square(tf.linalg.global_norm(model_difference))
    return squared_norm

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
    def training(self, client_images, client_labels, client_unlabels, client_model, global_model_weights, local_epoch, batch_size):

        client_model.set_weights(global_model_weights)
        
        data_num = client_images.shape[0]
        train_step = data_num//batch_size
            
        def weak_augment(image, flip, translation):
            image = flip(image)
            image = translation(image)
            return image

        translation = tf.keras.layers.RandomTranslation(translate_range , translate_range , fill_mode='constant', fill_value=0.5)
        flip = tf.keras.layers.RandomFlip("horizontal")
        
        rand_aug = iaa.RandAugment(n=randaug_n, m=randaug_m)
        
        
        for e in range(local_epoch):
            for s in range(train_step):
                with tf.GradientTape() as tape:
                    predictions = client_model(client_images[s*batch_size:min((s+1)*batch_size, len(client_images))], training=True)
                    loss_label = self.loss_fn(client_labels[s*batch_size:min((s+1)*batch_size, len(client_images))], predictions)
                    
                    batch_unlabel = client_unlabels[s * batch_size * mu_unlabel : min((s+1) * mu_unlabel * batch_size, len(client_unlabels))]
                    weak_unlabel = weak_augment(batch_unlabel, flip, translation)
                    weak_pred = client_model(weak_unlabel, training=True)

                    above_th = np.where(np.amax(weak_pred, axis=-1) > threshold)
                    pseudo_label = (tf.keras.utils.to_categorical(np.argmax(weak_pred, axis=1), num_class)[above_th])
                    
                    batch_unlabel = batch_unlabel[above_th]
                    strong_unlabel = rand_aug(images = (batch_unlabel * 255.0).astype(np.uint8))/255.0
                    strong_pred = client_model(strong_unlabel, training=True)


                    if len(strong_unlabel) > 0:
                        loss_unlabel = self.loss_fn(pseudo_label, strong_pred)
                    else:
                        loss_unlabel = 0.0

                    if loss_unlabel != 0.0:
                        print('use unlabel loss')
                        loss = loss_label + weight_unlabel * loss_unlabel
                    else:
                        loss = loss_label
                    if fl_framework == 'fedprox':
                        proxy = (mu_prox/2)*difference_model_norm_2_square(global_model_weights, client_model.get_weights())
                        loss += proxy
                grads = tape.gradient(loss, client_model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, client_model.trainable_weights))

        del global_model_weights

        return client_model.get_weights()#, hist



    
##########################################################################################################
server = Server(build_global_model(),
                RMSprop(learning_rate=lr),
                tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                val_images,
                val_labels,
                test_images,
                test_labels)


client_list = []

for c in range(num_client):
    client_list.append(
        Client(RMSprop(learning_rate=lr),
               tf.keras.losses.CategoricalCrossentropy(from_logits=False))
              ) 
    
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
            train_unlabels_list[client_idx[c]],
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
    
    

