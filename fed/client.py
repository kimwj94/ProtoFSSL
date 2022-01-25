import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
from utils import calc_euclidian_dists, get_prototype



class Client:
    def __init__(self, 
                optimizer, 
                s_label=1, 
                q_label=2,
                num_label=5,
                q_unlabel= 100,
                num_class=10, 
                local_episode=10, 
                input_shape=(32,32,3),
                unlabel_round=0,
                weight_unlabel=3e-1,
                unlabel_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
                num_round=300):
        self.optimizer = optimizer
        self.s_label = s_label
        self.q_label = q_label
        self.q_unlabel = q_unlabel
        self.num_class = num_class
        self.local_episode = local_episode
        self.unlabel_loss_fn = unlabel_loss_fn
        self.input_shape = input_shape
        self.unlabel_round = unlabel_round
        self.weight_unlabel = weight_unlabel
        self.num_label = num_label
        self.base_lr = copy.deepcopy(optimizer.lr.numpy())        
        self.num_round = num_round

    # training using global model weights
    # return: update client model weights
    def calc_proto(self, 
                client_dataset, 
                client_idx,
                client_model, 
                global_model_weights
                ):
        
        client_model.set_weights(global_model_weights)
        #K.set_value(self.optimizer.learning_rate, lr)
        temp_dataset = client_dataset[client_idx]
        set_label = []
        
        # sample labeled data
        for idx in range(self.num_class):
            label_idx = np.arange(self.num_label)
            set_label.append(np.take(temp_dataset[str(idx)], label_idx, axis=0))
            
        
        # transform to numpy array
        set_label = np.array(set_label)

        # reshape for input then concatenate
        set_label = np.reshape(set_label, (self.num_class * self.num_label,)+self.input_shape)

        # get embedding vector
        z = client_model(set_label, training=False)
        local_proto = get_prototype(z, self.num_label, self.num_class)

        return local_proto

    def set_learning_rate(self, curr_round):
        curr_lr = self.base_lr * np.cos(7 * np.pi * curr_round / (16 * self.num_round))        
        curr_lr = max(curr_lr, 1e-6)        
        K.set_value(self.optimizer.learning_rate, curr_lr)

    # training using global model weights
    # return: update client model weights
    def training(self, 
                client_dataset, 
                client_idx,
                client_model, 
                global_model_weights,
                client_protos,
                rounds):
        
        client_model.set_weights(global_model_weights)
        self.set_learning_rate(rounds)
        temp_dataset = client_dataset[client_idx]

        # local episode
        client_acc = 0.0
        client_loss = 0.0
        for e in range(self.local_episode):            
            support_set_label = []
            query_set_label = []
            query_set_unlabel = []

            # sample labeled data
            for idx in range(self.num_class):
                label_idx = np.random.choice(len(temp_dataset[str(idx)]), self.s_label+self.q_label, replace=False)
                support_set_label.append(np.take(temp_dataset[str(idx)], label_idx[:self.s_label], axis=0))
                query_set_label.append(np.take(temp_dataset[str(idx)], label_idx[self.s_label:], axis=0))
                
            # sample unlabeled data
            unlabel_idx = np.random.choice(len(temp_dataset['unlabel']), self.q_unlabel, replace=False)
            query_set_unlabel.append(np.take(temp_dataset['unlabel'], unlabel_idx, axis=0))
            
            # transform to numpy array
            support_set_label = np.array(support_set_label)
            query_set_label = np.array(query_set_label)
            query_set_unlabel = np.array(query_set_unlabel)

            # reshape for input then concatenate
            support_set_label = np.reshape(support_set_label, (self.num_class * self.s_label,)+self.input_shape)
            query_set_label = np.reshape(query_set_label, (self.num_class * self.q_label,)+self.input_shape)
            query_set_unlabel = np.reshape(query_set_unlabel, (self.q_unlabel,)+self.input_shape)
            
            # label for one-hot vector
            y = np.tile(np.arange(self.num_class)[:, np.newaxis], (1, self.q_label))
            y_onehot = tf.stop_gradient(tf.cast(tf.one_hot(y, self.num_class), tf.float32))
            
            cat = tf.concat([support_set_label, query_set_label, query_set_unlabel], axis=0)

            with tf.GradientTape() as tape:

                # get embedding vector
                z = client_model(cat, training=True)

                # make prototype
                prototype = get_prototype(z[:len(support_set_label)], self.s_label, self.num_class)

                # compute distance between query and prototype
                z_query = z[len(support_set_label):]
                q_dists = calc_euclidian_dists(z_query, prototype) # shape: (data 개수, 10)
                
                # compute loss (negative log probability)
                log_p_y = tf.nn.log_softmax(-q_dists[:len(query_set_label)], axis=-1)
                log_p_y = tf.reshape(log_p_y, [self.num_class, self.q_label, -1])
                loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
                
                # loss for unlabel data
                if rounds > self.unlabel_round:
                    p_y_unlabel = tf.nn.softmax(-q_dists[len(query_set_label):], axis=-1)
                    client_predictions = []
                    # make distribution using client's prototype
                    for i in range(len(client_protos)):
                        q_dists_client = calc_euclidian_dists(z_query[len(query_set_label):], client_protos[i])
                        p_y_unlabel_client = tf.nn.softmax(-q_dists_client, axis=-1)
                        client_predictions.append(p_y_unlabel_client)
                    
                    # average all distribution
                    client_p = tf.stack(client_predictions, axis=0)
                    averaged_p = tf.reduce_mean(client_p, axis=0)
                    
                    # sharpening
                    T = 0.5
                    sharpened_p = tf.pow(averaged_p, 1.0/T)
                    # normalize
                    normalized_p = sharpened_p / tf.reshape(tf.reduce_sum(sharpened_p, axis=1), (-1, 1))
                    

                    loss_unlabel = self.unlabel_loss_fn(normalized_p, p_y_unlabel)
                    loss += self.weight_unlabel * loss_unlabel
                        
                eq = tf.cast(tf.equal(
                        tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32), 
                        tf.cast(y, tf.int32)), tf.float32)   
                client_loss += loss
                acc = tf.reduce_mean(eq)
                client_acc += acc
            
            # compute gradient
            grads = tape.gradient(loss, client_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, client_model.trainable_weights))
        
                
        client_acc /= self.local_episode
        client_loss /= self.local_episode
        
        # calcuate local prototype to send to server
        support_set_label = []
        # use all labeled data
        for label in range(self.num_class):
            label_idx = np.array(list(range(self.num_label)))
            support_set_label.append(np.take(client_dataset[client_idx][str(label)], label_idx, axis=0))

        support_set_label = np.array(support_set_label)
        support_set_label = np.reshape(support_set_label, (-1,)+self.input_shape)

        z = client_model(support_set_label, training=False)
        local_proto = get_prototype(z, self.num_label, self.num_class)

        del global_model_weights
        del temp_dataset
        return client_model.get_weights(), local_proto, client_acc, client_loss