import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
from utils import calc_euclidian_dists, get_prototype

class Server:
    
    def __init__(self, 
                global_model,                
                val_dataset,
                test_dataset,
                num_class=10,
                input_shape=(32,32,3)):
        self.global_model = global_model
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_class = num_class
        
        self.client_model_weight_list = []
        self.client_prototype_list = []
        self.input_shape = input_shape
   
    # return: global model weights
    def get_global_model_weights(self):
        return self.global_model.get_weights()
    
    def get_client_prototype(self):
        return self.client_prototype_list
    
    def reset(self):
        self.client_model_weight_list = []
        self.client_prototype_list = []

    # input: client model weight
    # save client model weights
    def rec_client_model_weights(self, client_model_weights):
        self.client_model_weight_list.append(client_model_weights)
    
    # input: client prototype
    # save client prototype
    def rec_cleint_prototype(self, client_prototype):
        self.client_prototype_list.append(client_prototype)
        
    # FedAvg client weights   
    def fed_avg(self):
        global_weights = list()
        for weights_list_tuple in zip(*self.client_model_weight_list): 
            global_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        self.global_model.set_weights(global_weights)
        
    # FedAvg and evaluate global model with validation set
    def get_accuracy(self, test_dataset, model):
        class_idx = list(range(self.num_class))
        query_set_label = []

        for idx in class_idx:
            label_idx = list(range(300))
            query_set_label.append(np.take(test_dataset[idx], label_idx, axis=0))

        query_set_label = np.array(query_set_label)
        query_set_label = np.reshape(query_set_label, (self.num_class * 300,) + self.input_shape)

        y = np.tile(np.arange(self.num_class)[:, np.newaxis], (1, 300))
        y_onehot = tf.stop_gradient(tf.cast(tf.one_hot(y, self.num_class), tf.float32))


        cat = tf.concat([query_set_label], axis=0)
        z = model(cat, training=False)

        client_predictions = []
        for client_proto in self.client_prototype_list:
            q_dists_client = calc_euclidian_dists(z, client_proto)
            p_y_unlabel_client = tf.nn.softmax(-q_dists_client, axis=-1)

            client_predictions.append(p_y_unlabel_client)

        # average all distribution
        client_p = tf.stack(client_predictions, axis=0)
        averaged_p = tf.reduce_mean(client_p, axis=0)
        preds = np.argmax(tf.reshape(averaged_p, [self.num_class, 300, -1]), axis=-1)
        eq = np.equal(preds, y.astype(int)).astype(np.float32)

        loss = tf.keras.losses.SparseCategoricalCrossentropy()(np.reshape(y.astype(int), [-1]), averaged_p)
        acc = np.mean(eq)

        return acc, loss

    def test_accuracy(self, r):

        acc, loss = self.get_accuracy(self.test_dataset, self.global_model)

        print("-----Test Acc: {}, Test Loss: {}".format(acc, loss))
        #with open(path + '/' +exp+'_test_acc' , 'a+') as f:
        #    f.write("{},{},{}\n".format(r,acc, loss))
        return loss, acc
    
    def val_accuracy(self, r):

        acc, loss = self.get_accuracy(self.val_dataset, self.global_model)
          
        print("-----Val Acc: {}, Val Loss: {}".format(acc, loss))
        #with open(path + '/' +exp+'_val_acc' , 'a+') as f:
        #    f.write("{},{},{}\n".format(r,acc, loss))
        return loss, acc