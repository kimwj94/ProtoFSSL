

import os
import sys
import copy
import time
from datetime import datetime
import random
import argparse


from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


from fed.server import Server
from fed.client import Client
from nets.resnet import ResNet9, ResNet18, WideResNet28x2
from datasets import get_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='protofssl', help='Experiment name')
parser.add_argument('--dataset', default='cifar10', help='The name of the datset. One of [cifar10, svhn, stl10], default: cifar10')
parser.add_argument('--model', default='res9', help='Model type. One of [res9, res18, wres28x2]')
parser.add_argument('--bn_type', default=None, help='Batch normalization type one of [bn, sbn, gn], default: None')
parser.add_argument('--non_iid', action='store_true', help='Run non-iid distributed data')
parser.add_argument('--num_round', type=int, default=300, help='Number of training round, default: 300')
parser.add_argument('--num_label', type=int, default=5, help='Number of labeled data per client per class, default: 5')
parser.add_argument('--num_unlabel', type=int, default=490, help='Number of unlabeled data per client, default: 490')
parser.add_argument('--local_episode', type=int, default=10, help='Number of local episode, default: 10')
parser.add_argument('--unlabel_round', type=int, default=0, help='Starting training round to use unlabeled data(non-inclusive), default: 0')
parser.add_argument('--optimizer', default='rmsprop', help='Which optimizer to use(rmsprop, sgd, adam), default: rmsprop')
parser.add_argument('--opt_lr', type=float, default=1e-3, help='Learning rate for optimizer')
parser.add_argument('--opt_momentum', type=float, default=0, help='Momentum for optimizer')
parser.add_argument('--num_client', type=int, default=100, help='Number of clients, default: 100')
parser.add_argument('--num_active_client', type=int, default=5, help='Number of active clients, default: 5')
parser.add_argument('--unlabel_loss_type', default='CE', help='Loss type to train unlabeled data, default: CE')
parser.add_argument('--refine_at_begin', action='store_true', help='Prototype refinement is done at the beginning of the epoch')
parser.add_argument('--keep_proto_rounds', type=int, default=1, help='Number of old prototypes to keep, default: 1')
parser.add_argument('--warmup_episode', type=int, default=0, help='Warmup episode before using unlabeled data, default: 0')
parser.add_argument('--l2_factor', type=float, default=1e-4, help='L2 Regularization factor, default: 1e-4')
parser.add_argument('--is_sl', action='store_true', help='Whether to do supervised learning')


FLAGS = parser.parse_args()


# GPU setting
gpus = tf.config.experimental.list_physical_devices('GPU')
idx = 0
tf.config.experimental.set_visible_devices(gpus[idx], 'GPU')
tf.config.experimental.set_memory_growth(gpus[idx], True)

# path for accuracy & loss
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(BASE_DIR, 'result')
isExist = os.path.exists(result_path)

if not isExist:  
    os.makedirs(result_path)

########################## Hyperparameters ###########################

SEED_NUM=1001
S_LABEL = 1
Q_LABEL = 2
Q_UNLABEL = 100 # number of unlabeled data for query set

IS_IID = False if FLAGS.non_iid else True# distribution of unlabeled data

NUM_LABEL = FLAGS.num_label # number of labeled data per class for one client
NUM_UNLABEL = FLAGS.num_unlabel # number of unlabeled data for one client

NUM_ROUND = FLAGS.num_round
NUM_CLIENT = FLAGS.num_client
NUM_ACTIVE_CLIENT = FLAGS.num_active_client

LOCAL_EPISODE = FLAGS.local_episode # number of local episode

UNLABEL_ROUND = FLAGS.unlabel_round # from what round to use unlabeled data
UNLABEL_LOSS_TYPE = FLAGS.unlabel_loss_type # loss for unlabeled data. MSE or CE

OPT = FLAGS.optimizer # optimizer
KEEP_PROTO_ROUNDS = FLAGS.keep_proto_rounds
WARMUP_EPISODE = FLAGS.warmup_episode

# get model
def get_model(model_name='res9', input_shape=(32,32,3), l2_factor=1e-4, is_sl=False, num_classes=10):
    
    #Define downsample sizes
    if input_shape[0] == 32:
        pool_list = [2,2,2,4]
    elif input_shape[0] == 96:
        pool_list=[3,2,4,4]
    
    if model_name == 'res9':    
        model = ResNet9(input_shape=input_shape, bn=FLAGS.bn_type, pool_list=pool_list, l2_factor=l2_factor, is_sl=is_sl, num_classes=num_classes)
    elif model_name == 'res18':
        model = ResNet18(input_shape=input_shape, bn=FLAGS.bn_type, pool_list=pool_list, l2_factor=l2_factor, is_sl=is_sl, num_classes=num_classes)
    elif model_name == 'wres28x2':
        model = WideResNet28x2(input_shape=input_shape, bn=FLAGS.bn_type, pool_list=pool_list, is_sl=is_sl, num_classes=num_classes)

    dummy_in = tf.convert_to_tensor(np.random.random((1,) + input_shape))
    out = model(dummy_in) 
    return model

def write_record(record_list, suffix):
    with open(os.path.join(result_path, FLAGS.exp_name + suffix), 'w+') as f:
        for record in record_list:
            f.write(record)            
    f.close()

if __name__=='__main__':
    # Get starting time
    startTime = datetime.now()

    client_dataset, val_dataset, test_dataset, NUM_CLASS, INPUT_SHAPE, client_labels = get_dataset(dataset_name=FLAGS.dataset,
                                                                                    is_iid=IS_IID,
                                                                                    num_client=NUM_CLIENT,
                                                                                    num_label=NUM_LABEL,
                                                                                    num_unlabel=NUM_UNLABEL,
                                                                                    is_sl=FLAGS.is_sl)
    server_model = get_model(FLAGS.model, INPUT_SHAPE, l2_factor=FLAGS.l2_factor, is_sl=FLAGS.is_sl)
    print('Model built:', FLAGS.model)   
    print(server_model.summary())

    server = Server(server_model,
                    val_dataset,
                    test_dataset,
                    num_class=NUM_CLASS,
                    input_shape=INPUT_SHAPE,
                    num_active_client=NUM_ACTIVE_CLIENT,
                    keep_proto_rounds=KEEP_PROTO_ROUNDS,
                    is_sl=FLAGS.is_sl)

    client_list = []


    if OPT == 'rmsprop':        
        optim = RMSprop(learning_rate=FLAGS.opt_lr, momentum=FLAGS.opt_momentum)
    elif OPT == 'sgd':
        #lr, mom = 0.1, 0.7
        optim = SGD(learning_rate=FLAGS.opt_lr, momentum=FLAGS.opt_momentum)
    elif OPT == 'adam':        
        optim = Adam(lr=FLAGS.opt_lr)

    if UNLABEL_LOSS_TYPE =='MSE':
        unlabel_loss_fn = tf.keras.losses.MeanSquaredError()
    elif UNLABEL_LOSS_TYPE == 'CE':
        unlabel_loss_fn = tf.keras.losses.CategoricalCrossentropy()


    for c in range(NUM_CLIENT):    
        client_list.append(Client(optimizer=optim,
                                    s_label=S_LABEL,
                                    q_label=Q_LABEL,
                                    num_label=NUM_LABEL,
                                    q_unlabel=Q_UNLABEL,
                                    num_class=NUM_CLASS,
                                    local_episode=LOCAL_EPISODE,
                                    input_shape=INPUT_SHAPE,
                                    unlabel_round=UNLABEL_ROUND,
                                    unlabel_loss_fn=unlabel_loss_fn,
                                    num_round=NUM_ROUND,
                                    warmup_episode=WARMUP_EPISODE                                
                                    ))

    print("Training Start")
    max_val, max_test = 0.0, 0.0
    max_round = 0
    cycle = 1
    client_model = get_model(FLAGS.model, INPUT_SHAPE, l2_factor=FLAGS.l2_factor, is_sl=FLAGS.is_sl)

    train_record_list = []
    val_record_list = []
    test_record_list = []

    for r in range(NUM_ROUND):
        round_start = time.time()
        print("Round {}".format(r+1))    
                
        global_model_weights = copy.deepcopy(server.get_global_model_weights())
        server.reset_weight()
        #server.reset_prototype()
        
        random.seed(r+SEED_NUM)
        client_idx = random.sample(range(NUM_CLIENT), NUM_ACTIVE_CLIENT)

        if FLAGS.refine_at_begin:
            # for each client
            for c in range(NUM_ACTIVE_CLIENT):
                client_prototype = client_list[client_idx[c]].calc_proto(
                                                        client_dataset,
                                                        client_idx[c],
                                                        client_model,
                                                        copy.deepcopy(global_model_weights))

            server.rec_cleint_prototype(client_prototype)
        
        total_client_acc = 0.0
        total_client_loss = 0.0

        if FLAGS.is_sl:
             # for each client
            for c in range(NUM_ACTIVE_CLIENT):      
                # training with global model 
                client_weight, client_acc, client_loss \
                    = client_list[client_idx[c]].supervised_training(
                                                        client_dataset,
                                                        client_labels,
                                                        client_idx[c],
                                                        client_model,
                                                        copy.deepcopy(global_model_weights),                                                        
                                                        r)
                
                total_client_acc += client_acc
                total_client_loss += client_loss

                # server receive client weights and prototypes
                server.rec_client_model_weights(client_weight)
        else:
            #Remove old prototypes
            server.update_client_prototypes()                   

            # get global model weights & prototypes of other clients
            client_protos = copy.deepcopy(server.get_client_prototype())
            # for each client
            for c in range(NUM_ACTIVE_CLIENT):      
                # training with global model 
                client_weight, client_prototype, client_acc, client_loss \
                    = client_list[client_idx[c]].training(
                                                        client_dataset,
                                                        client_idx[c],
                                                        client_model,
                                                        copy.deepcopy(global_model_weights),
                                                        client_protos,
                                                        r)
                
                total_client_acc += client_acc
                total_client_loss += client_loss

                # server receive client weights and prototypes
                server.rec_client_model_weights(client_weight)
                if not FLAGS.refine_at_begin:
                    server.rec_cleint_prototype(client_prototype)

        total_client_acc /= NUM_ACTIVE_CLIENT
        total_client_loss /= NUM_ACTIVE_CLIENT
        
        # FedAvg 
        server.fed_avg()

        print("--Training acc: {}, loss: {}".format(total_client_acc, total_client_loss))
        train_record_list.append("{},{},{}\n".format(r+1,total_client_acc, total_client_loss))
        
        #with open(path + '/' +exp+'_train_acc' , 'a+') as f:
        #    f.write("{},{},{}\n".format(r+1,total_client_acc, total_client_loss))

        # val & test accuracy
        val_loss, val_acc = server.val_accuracy(r+1)
        test_loss, test_acc = server.test_accuracy(r+1) 
        
        if val_acc > max_val:
            max_val = val_acc
            max_test = test_acc
            max_round = r+1

        val_record_list.append("{},{},{}\n".format(r+1,val_acc, val_loss))
        test_record_list.append("{},{},{}\n".format(r+1,test_acc, test_loss))

        round_end = time.time()
        print("--Time for Round {}: {}".format(r, round_end-round_start))
    
    dt_string = startTime.strftime("%Y%m%d %H%M")

    # Write record
    write_record(train_record_list, '_' + dt_string + '_train_acc')
    write_record(val_record_list, '_' + dt_string + '_val_acc')
    write_record(test_record_list, '_' + dt_string + '_test_acc')

    with open(os.path.join(result_path, 'summary'), 'a+') as f: 
        if FLAGS.bn_type is None:
            model_name = FLAGS.model
        else:
            model_name = FLAGS.model + FLAGS.bn_type
        f.write("expname: {}, time: {}, dataset: {}, model: {}, maxround: {}, maxval: {:.6f}, maxtest: {:.6f}\n"\
            .format(FLAGS.exp_name, dt_string, FLAGS.dataset, model_name, max_round, max_val, max_test))
    f.close()
    print("Max test accuracy: ", max_test)


