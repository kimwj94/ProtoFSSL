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
                input_shape=(32,32,3),
                num_active_client=5,
                keep_proto_rounds=1,
                is_sl=False,
                print_log=True):
        self.global_model = global_model
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_class = num_class
        
        self.client_model_weight_list = []
        self.client_prototype_list = []
        self.input_shape = input_shape
        self.num_active_client = num_active_client
        self.keep_proto_rounds = keep_proto_rounds
        self.is_sl = is_sl
        self.print_log = print_log
   
    # return: global model weights
    def get_global_model_weights(self):
        return self.global_model.get_weights()
    
    def get_client_prototype(self):
        return self.client_prototype_list
    
    def comp_dist(self):
        dist_list = [[] for _ in range(10)]
        dist_avg_list = []
        print("comd dist for {} clients".format(len(self.client_prototype_list[self.num_active_client:])))
        for label in range(10):
            dist_sum = 0.0
            dist_cnt = 0.0
            for i in range(len(self.client_prototype_list[self.num_active_client:])):
                for j in range(i+1, len(self.client_prototype_list[self.num_active_client:])):
                    proto1 = self.client_prototype_list[self.num_active_client:][i][label]
                    proto2 = self.client_prototype_list[self.num_active_client:][j][label]
                    dist = tf.math.pow(tf.reduce_sum(tf.math.pow(proto1-proto2, 2)), 0.5)
                    dist_sum += dist
                    dist_cnt += 1.0
                    dist_list[label].append(dist.numpy())
            
            if dist_cnt != 0:
                dist_avg = dist_sum/dist_cnt
                dist_avg_list.append(dist_avg.numpy())
        
        return self.client_prototype_list, dist_list, dist_avg_list
                    
                    #tf.math.pow(tf.reduce_sum(tf.math.pow(x - y, 2), 2), 0.5)
        
    
    def reset_weight(self):
        self.client_model_weight_list = []
    
    def reset_prototype(self):
        self.client_prototype_list = []

    def update_client_prototypes(self):
        if len(self.client_prototype_list) > self.num_active_client * self.keep_proto_rounds:
            self.client_prototype_list = self.client_prototype_list[-self.num_active_client * self.keep_proto_rounds:]

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
                #with open(path + '/' +exp+'_train_acc' , 'a+') as f:
        #    f.write("{},{},{}\n".format(r+1,total_client_acc, total_client_loss))
        class_idx = list(range(self.num_class))
        
        per_class = len(test_dataset[0])

        final_acc = 0.0
        final_loss = 0.0

        total_num = per_class * self.num_class        
        div = 10
        per_class_div = per_class // div
        for i in range(div):
            query_set_label = []

            for idx in class_idx:
                label_idx = list(range(i*per_class_div, (i+1)*per_class_div))
                query_set_label.append(np.take(test_dataset[idx], label_idx, axis=0))

            query_set_label = np.array(query_set_label)
            query_set_label = np.reshape(query_set_label, (self.num_class * per_class_div,) + self.input_shape)

            y = np.tile(np.arange(self.num_class)[:, np.newaxis], (1, per_class_div))
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
            preds = np.argmax(tf.reshape(averaged_p, [self.num_class, per_class_div, -1]), axis=-1)
            eq = np.equal(preds, y.astype(int)).astype(np.float32)

            loss = tf.keras.losses.SparseCategoricalCrossentropy()(np.reshape(y.astype(int), [-1]), averaged_p)
            acc = np.mean(eq)

            final_acc += acc
            final_loss += loss

        final_acc /= div
        final_loss /= div

        return final_acc, final_loss

    # FedAvg and evaluate global model with validation set
    def get_accuracy_sl(self, test_dataset, model):

        class_idx = list(range(self.num_class))
        
        per_class = len(test_dataset[0])

        final_acc = 0.0
        final_loss = 0.0

        total_num = per_class * self.num_class        
        div = 10
        per_class_div = per_class // div
        for i in range(div):
            iamges = []

            for idx in class_idx:
                label_idx = list(range(i*per_class_div, (i+1)*per_class_div))
                iamges.append(np.take(test_dataset[idx], label_idx, axis=0))

            iamges = np.array(iamges)
            iamges = np.reshape(iamges, (self.num_class * per_class_div,) + self.input_shape)

            labels = np.tile(np.arange(self.num_class)[:, np.newaxis], (1, per_class_div))
            labels = np.reshape(labels.astype(int), [-1])

            z = model(iamges, training=False)

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            loss = loss_fn(labels, z).numpy()
            
            eq = tf.cast(tf.equal(
                        tf.cast(tf.argmax(z, axis=-1), tf.int32), 
                        tf.cast(labels, tf.int32)), tf.float32)               
            acc = tf.reduce_mean(eq)
            final_acc += acc
            final_loss += loss

        final_acc /= div
        final_loss /= div
        
        return final_acc, final_loss

    def test_accuracy(self, r):

        if self.is_sl:
            acc, loss = self.get_accuracy_sl(self.test_dataset, self.global_model)
        else:
            acc, loss = self.get_accuracy(self.test_dataset, self.global_model)
        if self.print_log:
            print("-----Test Acc: {}, Test Loss: {}".format(acc, loss))
        #with open(path + '/' +exp+'_test_acc' , 'a+') as f:
        #    f.write("{},{},{}\n".format(r,acc, loss))
        return loss, acc
    
    def val_accuracy(self, r):

        if self.is_sl:
            acc, loss = self.get_accuracy_sl(self.val_dataset, self.global_model)
        else:
            acc, loss = self.get_accuracy(self.val_dataset, self.global_model)
        if self.print_log:
            print("-----Val Acc: {}, Val Loss: {}".format(acc, loss))
        #with open(path + '/' +exp+'_val_acc' , 'a+') as f:
        #    f.write("{},{},{}\n".format(r,acc, loss))
        return loss, acc