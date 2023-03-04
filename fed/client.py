import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import math
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
from utils import calc_euclidian_dists, get_prototype, get_prototype2, difference_model_norm_2_square, calc_cosine_sim

from scipy.ndimage.interpolation import rotate, shift


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
                num_round=300,
                warmup_episode=0,
                sl_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                fl_framework='fedavg',
                mu=None):
        
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
        self.warmup_episode = warmup_episode
        self.sl_loss_fn = sl_loss_fn
        self.fl_framework=fl_framework
        self.mu=mu

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
        set_label, y = [], []
        
        # sample labeled data
        # for idx in range(self.num_class):
        #     label_idx = np.arange(self.num_label)
        #     set_label.append(np.take(temp_dataset[str(idx)], label_idx, axis=0))

        # sample labeled data
        for idx in range(self.num_class):
            if len(temp_dataset[str(idx)]) > 0:                
                set_label.append(temp_dataset[str(idx)])
                y.extend([idx]*len(temp_dataset[str(idx)]))               
        
        # transform to numpy array
        set_label = np.vstack(set_label)
        y = np.array(y)

        # reshape for input then concatenate
        set_label = np.reshape(set_label, (self.num_class * self.num_label,)+self.input_shape)

        # get embedding vector
        z = client_model(set_label, training=False)
        # local_proto = get_prototype(z, self.num_label, self.num_class)
        local_proto = get_prototype2(z, y, self.num_class)

        return local_proto

    def set_learning_rate(self, curr_round):
        # curr_lr = self.base_lr * np.cos(7 * np.pi * curr_round / (16 * self.num_round))                
        curr_lr = self.base_lr * np.cos(15 * np.pi * curr_round / (32 * self.num_round))                
        curr_lr = max(curr_lr, 1e-6)        
        K.set_value(self.optimizer.learning_rate, curr_lr)

    def augment(self, dataset, rounds):
        for key in dataset:
            # if key == 'unlabel':                
            #     continue
            images = dataset[key]
            # Flip lr
            np.random.seed(rounds)
            sampled = np.random.choice(len(images), int(len(images) * 0.25), replace=False)
            images[sampled] = np.flip(images[sampled], axis=-2)
            
            # Flip ud
            sampled = np.random.choice(len(images), int(len(images) * 0.25), replace=False)
            images[sampled] = np.flip(images[sampled], axis=-3)
            
            # Random shifts
            images = np.array([shift(img, [np.random.randint(-2, 2), np.random.randint(-2, 2), 0]) for img in images]) # random shift
            dataset[key] = images
        return dataset

    # training using global model weights
    # return: update client model weights
    def training(self, 
                client_dataset, 
                client_idx,
                client_model, 
                global_model_weights,
                client_protos,
                rounds,
                use_noise,
                stddev):
        
        client_model.set_weights(global_model_weights)
        # self.set_learning_rate(rounds)
        temp_dataset = client_dataset[client_idx]

        from imgaug import augmenters as iaa
        randaug_n, randaug_m = 1, 14
        rand_aug = iaa.RandAugment(n=randaug_n, m=randaug_m)

        # temp_dataset = self.augment(temp_dataset, rounds)
        loss_method = 'cosine'

        label_data_dist = np.zeros(self.num_class)
        total_num_data = 0

        for idx in range(self.num_class):
            label_data_dist[idx] = len(temp_dataset[str(idx)])
            total_num_data += len(temp_dataset[str(idx)])

        label_data_dist /= total_num_data        

        # local episode
        client_acc = 0.0
        client_loss = 0.0
        client_loss_unlabel = 0.0

        for e in range(self.local_episode):            
            support_set_label = []
            query_set_label = []
            query_set_unlabel = []
            
            y_s = []
            y_q = []

            # sample labeled data
            for idx in range(self.num_class):
                if len(temp_dataset[str(idx)]) > 0:
                    label_idx = np.random.choice(len(temp_dataset[str(idx)]), (self.s_label + self.q_label), replace=False)
                    support_set_label.append(np.take(temp_dataset[str(idx)], label_idx[:self.s_label], axis=0))
                    query_set_label.append(np.take(temp_dataset[str(idx)], label_idx[self.s_label:], axis=0))
                    y_s.extend([idx]*self.s_label)
                    y_q.extend([idx]*self.q_label)
                    
            num_unlabel = len(temp_dataset['unlabel'])
            num_unlabel_cls = num_unlabel // 10
            # sample unlabeled data
            if rounds > self.unlabel_round and e >= self.warmup_episode:   
                # q_unlabel_train = max(1, int(self.q_unlabel * rounds / 300))
                q_unlabel_train = self.q_unlabel
                unlabel_idx = np.random.choice(len(temp_dataset['unlabel']), int(q_unlabel_train), replace=False)
                # unlabel_labels = [i for i in range(10) for _ in range(num_unlabel_cls)]                
                # unlabel_labels = np.take(unlabel_labels, unlabel_idx, axis=0)                
                query_set_unlabel.append(np.take(temp_dataset['unlabel'], unlabel_idx, axis=0))
                query_set_unlabel = np.vstack(query_set_unlabel) #/ 255.0
                query_set_unlabel = np.reshape(query_set_unlabel, (q_unlabel_train,)+self.input_shape)
            
            # transform to numpy array
            support_set_label = np.vstack(support_set_label) #/ 255.0
            query_set_label = np.vstack(query_set_label) #/ 255.0            
            
            y_s = np.array(y_s)
            y_q = np.array(y_q)

            # reshape for input then concatenate
            # support_set_label = np.reshape(support_set_label, (self.num_class * self.s_label,)+self.input_shape)
            # query_set_label = np.reshape(query_set_label, (self.num_class * self.q_label,)+self.input_shape)
            support_set_label = np.reshape(support_set_label, (-1,)+self.input_shape)
            query_set_label = np.reshape(query_set_label, (-1,)+self.input_shape)

            support_set_label = rand_aug(images = (support_set_label * 255.0).astype(np.uint8))/255.0
            
            
            # label for one-hot vector
            # y = np.tile(np.arange(self.num_class)[:, np.newaxis], (1, self.q_label))
            y_onehot = tf.stop_gradient(tf.cast(tf.one_hot(y_q, self.num_class), tf.float32)) # shape: (q_label * num_class, num_class)
            
            # sample unlabeled data
            if rounds > self.unlabel_round and e >= self.warmup_episode:                
                cat = tf.concat([support_set_label, query_set_label, query_set_unlabel], axis=0)
            else:
                cat = tf.concat([support_set_label, query_set_label], axis=0) 

            with tf.GradientTape() as tape:

                # get embedding vector
                z = client_model(cat, training=True)                                                
                z /= tf.norm(z, axis=-1, keepdims=True)                 

                # make prototype
                # prototype = get_prototype(z[:len(support_set_label)], self.s_label, self.num_class)
                prototype = get_prototype2(z[:len(support_set_label)], y_s, self.num_class)                

                # compute distance between query and prototype
                z_query = z[len(support_set_label):]
                if loss_method == 'euclidean':
                    q_dists = -calc_euclidian_dists(z_query, prototype) # shape: (q_label * num_class, num_class)
                else:
                    q_dists = calc_cosine_sim(z_query, prototype) / 0.1 # shape: (q_label * num_class, num_class)
                
                # compute loss (negative log probability)
                q_dists_label = q_dists[:len(query_set_label)]
                q_dists_unlabel = q_dists[len(query_set_label):]
                log_p_y = tf.nn.log_softmax(q_dists_label, axis=-1) # shape: (q_label * num_class, num_class)

                # log_p_y = tf.reshape(log_p_y, [self.num_class, self.q_label, -1])
                loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
                
                # loss for unlabel data
                if rounds > self.unlabel_round and e >= self.warmup_episode:                    
                    p_y_unlabel = tf.nn.softmax(q_dists_unlabel, axis=-1) #(q_unlabel*num_class, num_class)

                    # Prepare pseudo labels
                    client_predictions = []
                    valid_tbl = tf.ones((len(client_protos), self.num_class), dtype=tf.float32)

                    client_predictions2 = []

                    z_query_unlabel = z_query[len(query_set_label):]
                    # make distribution using client's prototype
                    for i in range(len(client_protos)):

                        if loss_method == 'euclidean':
                            q_dists_client = -calc_euclidian_dists(z_query_unlabel, client_protos[i])                            
                        else:
                            q_dists_client = calc_cosine_sim(z_query_unlabel, client_protos[i]) / 0.1                            

                        p_y_unlabel_client = tf.nn.softmax(q_dists_client, axis=-1)
                        client_predictions.append(p_y_unlabel_client)     

                        strong_unlabel = rand_aug(images = (query_set_unlabel * 255.0).astype(np.uint8))/255.0
                        strong_pred = client_model(strong_unlabel, training=True)         

                        if loss_method == 'euclidean':                            
                            q_dists_client_strong = -calc_euclidian_dists(strong_pred, client_protos[i])
                            q_dists_strong = -calc_euclidian_dists(strong_pred, prototype)
                        else:                            
                            q_dists_client_strong = calc_cosine_sim(strong_pred, client_protos[i]) / 0.1
                            q_dists_strong = calc_cosine_sim(strong_pred, prototype) / 0.1                

                        p_y_unlabel_client2 = tf.nn.softmax(q_dists_client_strong, axis=-1)
                        client_predictions2.append(p_y_unlabel_client2)   
                        p_y_unlabel2 = tf.nn.softmax(q_dists_strong, axis=-1)

                    
                    # average all distribution
                    client_p = tf.stack(client_predictions, axis=0) # (num_client, q_unlabel, num_class)                    
                    averaged_p = tf.reduce_mean(client_p, axis=0)
                    
                    # sharpening
                    T = 0.5
                    sharpened_p = tf.pow(averaged_p, 1.0/T)
                    
                    # normalize
                    normalized_p = sharpened_p / tf.reshape(tf.reduce_sum(sharpened_p, axis=1), (-1, 1))

                    # average all distribution
                    client_p2 = tf.stack(client_predictions2, axis=0) # (num_client, q_unlabel, num_class)                    
                    averaged_p2 = tf.reduce_mean(client_p2, axis=0)
                    
                    # sharpening
                    T = 0.5
                    sharpened_p2 = tf.pow(averaged_p2, 1.0/T)
                    
                    # normalize
                    normalized_p2 = sharpened_p2 / tf.reshape(tf.reduce_sum(sharpened_p2, axis=1), (-1, 1))

                    # loss_unlabel = self.unlabel_loss_fn(normalized_p2, p_y_unlabel2)
                    loss_unlabel = (self.unlabel_loss_fn(normalized_p2, p_y_unlabel) + self.unlabel_loss_fn(normalized_p, p_y_unlabel2)) / 2

                    # loss_unlabel = self.unlabel_loss_fn(normalized_p, p_y_unlabel)
                    # p_y_unlabel = tf.nn.log_softmax(-q_dists_unlabel, axis=-1) #(q_unlabel*num_class, num_class)
                    # loss_unlabel = tf.sort(-tf.reduce_sum(tf.multiply(p_y_unlabel, normalized_p), axis=-1))[:int((q_unlabel_train+1) // 2)]                    
                    # loss_unlabel = tf.reduce_mean(loss_unlabel)

                    client_loss_unlabel += loss_unlabel * self.weight_unlabel
                    loss += self.weight_unlabel * loss_unlabel
                   
                if self.fl_framework == 'fedprox':
                    proxy = (self.mu/2)*difference_model_norm_2_square(global_model_weights, client_model.get_weights())
                    loss += proxy
                    
                eq = tf.cast(tf.equal(
                        tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32), 
                        tf.cast(y_q, tf.int32)), tf.float32)   
                client_loss += loss
                acc = tf.reduce_mean(eq)
                client_acc += acc
            
            # compute gradient
            grads = tape.gradient(loss, client_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, client_model.trainable_weights))
        
                
        client_acc /= self.local_episode
        client_loss /= self.local_episode
        client_loss_unlabel /= self.local_episode        
        
        # calcuate local prototype to send to server
        support_set_label = []
        # use all labeled data
        # for label in range(self.num_class):
        #     label_idx = np.array(list(range(self.num_label)))
        #     support_set_label.append(np.take(temp_dataset[str(label)], label_idx, axis=0))

        # sample labeled data
        y_s = []
        for idx in range(self.num_class):
            if len(temp_dataset[str(idx)]) > 0:
                scale = label_data_dist[idx] * self.num_class                
                support_set_label.append(temp_dataset[str(idx)])
                y_s.extend([idx]*len(temp_dataset[str(idx)]))                

        # support_set_label = np.array(support_set_label) #/255.0
        support_set_label = np.vstack(support_set_label)        
        support_set_label = np.reshape(support_set_label, (-1,)+self.input_shape)
        y_s = np.array(y_s)

        z = client_model(support_set_label, training=False)
        z /= tf.norm(z, axis=-1, keepdims=True)         
        
        # local_proto1 = get_prototype(z, self.num_label, self.num_class, use_noise, stddev)
        local_proto = get_prototype2(z, y_s, self.num_class, use_noise, stddev)

        # if rounds > self.unlabel_round:
        #     unlabel_eq = tf.cast(tf.equal(
        #                     tf.cast(tf.argmax(p_y_unlabel, axis=-1), tf.int32), 
        #                     tf.cast(unlabel_labels, tf.int32)), tf.float32) 
        #     print("unlabel acc", tf.reduce_mean(unlabel_eq).numpy())

        #     unlabel_eq = tf.cast(tf.equal( 
        #                     tf.cast(tf.argmax(normalized_p, axis=-1), tf.int32), 
        #                     tf.cast(unlabel_labels, tf.int32)), tf.float32) 
        #     print("unlabel Pseudo-label acc", tf.reduce_mean(unlabel_eq).numpy())           

            

        #     # argsort_idx = tf.where(tf.reduce_max(normalized_p, -1) > 0.9)
        #     z_unlabel = client_model(temp_dataset['unlabel'])
        #     if loss_method == 'euclidean':
        #         q_dists_unlabel = - calc_euclidian_dists(z_unlabel, client_protos[0])
        #     else:
        #         q_dists_unlabel = calc_cosine_sim(z_unlabel, client_protos[0]) / 0.1
        #     p_y_unlabel = tf.nn.softmax(q_dists_client, axis=-1)
        #     argsort_idx = tf.argsort(-tf.reduce_max(normalized_p, -1))[:int(num_unlabel * 0.1 * rounds / 300)]
            
        #     p_top10 = tf.gather(normalized_p, indices=argsort_idx)
        #     labels_top10 = tf.gather(unlabel_labels, indices=argsort_idx)
        #     unlabel_eq_top10 = tf.cast(tf.equal(
        #                     tf.cast(tf.argmax(p_top10, axis=-1), tf.int32), 
        #                     tf.cast(labels_top10, tf.int32)), tf.float32) 
        #     print("unlabel Pseudo-label acc top 10", tf.reduce_mean(unlabel_eq_top10).numpy())
        #     # unlabel_top10 = tf.gather(z_query_unlabel, indices=argsort_idx)
        #     # pseudo_labels_top10 = tf.argmax(p_top10, axis=-1)
        #     # local_proto_unlabel = get_prototype2(unlabel_top10, pseudo_labels_top10, self.num_class, use_noise, stddev, client_protos[0])
            
        #     consist_eq = tf.equal( 
        #                     tf.cast(tf.argmax(normalized_p, axis=-1), tf.int32), 
        #                     tf.cast(tf.argmax(normalized_p2, axis=-1), tf.int32))            
            
        #     print("unlabel consistency", np.mean(consist_eq.numpy())) 
        #     if np.mean(consist_eq.numpy()) > 0:
        #         z_unlabel = client_model(query_set_unlabel, training=False)
        #         z_unlabel /= tf.norm(z_unlabel, axis=-1, keepdims=True)    

        #         consist_idx = tf.where(consist_eq)[0]            
        #         p_consist = tf.gather(z_unlabel, indices=consist_idx)
        #         pseudo_labels_consist = tf.argmax(p_consist, axis=-1)            
        #         local_proto_unlabel = get_prototype2(p_consist, pseudo_labels_consist, self.num_class, use_noise, stddev, client_protos[0])

        #         # unlabel_t = 0.5 * rounds / 300
        #         unlabel_t = 0.3
        #         # local_proto = local_proto * (1 - unlabel_t) + local_proto_unlabel * unlabel_t
        
        if len(client_protos) > 0:
            local_proto = client_protos[0] * 0.7 + local_proto * 0.3


        del global_model_weights
        del temp_dataset
        return client_model.get_weights(), local_proto, client_acc, client_loss, client_loss_unlabel


    # training using global model weights
    # return: update client model weights
    def supervised_training(self, 
                client_dataset, 
                client_labels,
                client_idx,
                client_model, 
                global_model_weights,                
                rounds):


        client_model.set_weights(global_model_weights)
        self.set_learning_rate(rounds)
        temp_dataset = client_dataset[client_idx]
        temp_labels = client_labels[client_idx]
        #temp_dataset = self.augment(temp_dataset, rounds)

        # local episode
        client_acc = 0.0
        client_loss = 0.0

        batch_size = 32
        train_step = math.ceil((self.num_label * self.num_class) / batch_size)

        images = []
        labels = []            

        # sample labeled data
        for idx in range(self.num_class):
            if len(temp_dataset[str(idx)]) > 0:
                images.append(temp_dataset[str(idx)])
                labels.append(temp_labels[str(idx)])

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)        
             
        images = np.reshape(images, (-1,) + self.input_shape)
        labels = np.reshape(labels, (-1))        

        np.random.seed(rounds)  
        idx = np.random.permutation(len(images))
        images = images[idx]
        labels = labels[idx]


        for e in range(self.local_episode):            
            # for s in range(train_step):
            with tf.GradientTape() as tape:
                batch_labels = labels
                batch_images = images
                predictions = client_model(batch_images, training=True)
                loss = self.sl_loss_fn(batch_labels, predictions)

                if self.fl_framework == 'fedprox':
                    proxy = (self.mu/2)*difference_model_norm_2_square(global_model_weights, client_model.get_weights())
                    loss += proxy
            
            grads = tape.gradient(loss, client_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, client_model.trainable_weights))

            eq = tf.cast(tf.equal(
                    tf.cast(tf.argmax(predictions, axis=-1), tf.int32), 
                    tf.cast(batch_labels, tf.int32)), tf.float32) 

            client_loss += loss * batch_labels.shape[0]
            acc = tf.reduce_sum(eq)
            client_acc += acc

        del global_model_weights

        client_acc /= (self.local_episode * images.shape[0])
        client_loss /= (self.local_episode * images.shape[0])

        return client_model.get_weights(), client_acc, client_loss
    

    # training using global model weights
    # return: update client model weights
    def fixmatch_training(self, 
                client_dataset, 
                client_idx,
                client_model, 
                global_model_weights,                
                rounds,
                use_noise=False,
                stddev=0):


        # Some hyperparameters
        batch_size = 10
        translate_range = (-0.125, 0.125)
        randaug_n = 2 # number of transformation 
        randaug_m = 14 # transformation range
        mu_unlabel = 10
        threshold = 0.95
        weight_unlabel = 1e-2

        client_model.set_weights(global_model_weights)
        self.set_learning_rate(rounds)
        temp_dataset = client_dataset[client_idx]

        images_l, labels_l = [], []
        total_num_data = 0

        for idx in range(self.num_class):
            images_l.extend(temp_dataset[str(idx)])            
            labels_l.extend([idx] * len(temp_dataset[str(idx)]))
            total_num_data += len(temp_dataset[str(idx)])
        
        train_step = total_num_data//batch_size
        images_l = np.stack(images_l)
        labels_l = np.stack(labels_l)        
        rand_idx = np.arange(len(images_l))
        np.random.shuffle(rand_idx)
        images_l, labels_l = images_l[rand_idx], labels_l[rand_idx]        
        
        
        client_unlabels = temp_dataset['unlabel']

        # local episode
        client_acc = 0.0
        client_loss = 0.0
        client_loss_unlabel = 0.0

        def weak_augment(image, flip, translation):
            image = flip(image)
            image = translation(image)
            return image

        # translation = tf.keras.layers.RandomTranslation(translate_range , translate_range , fill_mode='constant', fill_value=0.5)
        translation = tf.keras.layers.experimental.preprocessing.RandomTranslation(translate_range , translate_range , fill_mode='constant', fill_value=0.5)
        # flip = tf.keras.layers.RandomFlip("horizontal")
        flip = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
        
        from imgaug import augmenters as iaa
        rand_aug = iaa.RandAugment(n=randaug_n, m=randaug_m)

        for e in range(self.local_episode): 
            with tf.GradientTape() as tape:
                predictions = client_model(images_l, training=True)
                batch_labels = labels_l
                loss_label = self.sl_loss_fn(batch_labels, predictions)
                
                batch_unlabel = client_unlabels
                weak_unlabel = weak_augment(batch_unlabel, flip, translation)
                weak_pred = client_model(weak_unlabel, training=True)

                above_th = np.where(np.amax(weak_pred, axis=-1) > threshold)
                # pseudo_label = (tf.keras.utils.to_categorical(np.argmax(weak_pred, axis=1), self.num_class)[above_th])
                pseudo_label = np.argmax(weak_pred, axis=1)[above_th]
                
                batch_unlabel = batch_unlabel[above_th]
                strong_unlabel = rand_aug(images = (batch_unlabel * 255.0).astype(np.uint8))/255.0
                strong_pred = client_model(strong_unlabel, training=True)


                if len(strong_unlabel) > 0:
                    loss_unlabel = self.sl_loss_fn(pseudo_label, strong_pred)
                else:
                    loss_unlabel = 0.0

                if loss_unlabel != 0.0:
                    # print('use unlabel loss')
                    loss = loss_label + weight_unlabel * loss_unlabel
                else:
                    loss = loss_label
                if self.fl_framework == 'fedprox':
                    proxy = (self.mu/2)*difference_model_norm_2_square(global_model_weights, client_model.get_weights())
                    loss += proxy
            grads = tape.gradient(loss, client_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, client_model.trainable_weights))

            eq = tf.cast(tf.equal(
                    tf.cast(tf.argmax(predictions, axis=-1), tf.int32), 
                    tf.cast(batch_labels, tf.int32)), tf.float32) 

            client_loss += loss * batch_labels.shape[0]
            acc = tf.reduce_sum(eq)
            client_acc += acc

        client_acc /= (self.local_episode * images_l.shape[0])
        client_loss /= (self.local_episode * images_l.shape[0])

        del global_model_weights
        del temp_dataset

        return client_model.get_weights(), client_acc, client_loss