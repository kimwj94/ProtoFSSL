import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
import tensorflow_datasets as tfds
import copy
import random

SEED_NUM = 101
# get train/validation/test dataset
def get_cifar10_dataset():

    cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
    cifar10_std = np.array([0.2471, 0.2435, 0.2616])
    
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    num_class = 10
    input_shape = (32,32,3)

    np.random.seed(SEED_NUM)
    idx = np.random.permutation(len(train_images))
    train_images = train_images[idx]
    train_labels = train_labels[idx]
    np.random.seed(SEED_NUM)
    idx = np.random.permutation(len(test_images))
    test_images = test_images[idx]
    test_labels = test_labels[idx]

    images = np.concatenate((train_images, test_images), axis = 0)
    labels = np.concatenate((train_labels, test_labels), axis = 0)

    images = images.astype('float32')
    labels = tf.keras.utils.to_categorical(labels, num_class)
   

    train_dataset = []
    val_dataset = []
    test_dataset = []
    for i in range(num_class):
        train_dataset.append(images[np.where(labels == 1)[1]==i][:5400, :, :, :]/255.0)
        val_dataset.append(images[np.where(labels == 1)[1]==i][5400:5700, :, :, :]/255.0)
        test_dataset.append(images[np.where(labels == 1)[1]==i][5700:, :, :, :]/255.0)
    
    return train_dataset, val_dataset, test_dataset, num_class, input_shape

# get train/validation/test dataset
def get_cifar100_dataset():

    (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

    num_class = 100
    input_shape = (32,32,3)

    np.random.seed(SEED_NUM)
    idx = np.random.permutation(len(train_images))
    train_images = train_images[idx]
    train_labels = train_labels[idx]
    np.random.seed(SEED_NUM)
    idx = np.random.permutation(len(test_images))
    test_images = test_images[idx]
    test_labels = test_labels[idx]

    images = np.concatenate((train_images, test_images), axis = 0)
    labels = np.concatenate((train_labels, test_labels), axis = 0)

    images = images.astype('float32')
    labels = tf.keras.utils.to_categorical(labels, num_class)
   

    train_dataset = []
    val_dataset = []
    test_dataset = []
    for i in range(num_class):
        train_dataset.append(images[np.where(labels == 1)[1]==i][:540, :, :, :]/255.0)
        val_dataset.append(images[np.where(labels == 1)[1]==i][540:570, :, :, :]/255.0)
        test_dataset.append(images[np.where(labels == 1)[1]==i][570:, :, :, :]/255.0)
    
    return train_dataset, val_dataset, test_dataset, num_class, input_shape


# get train/validation/test dataset
def get_svhn_dataset():
    train_dataset = tfds.load(name="svhn_cropped", split=tfds.Split.TRAIN)
    test_dataset = tfds.load(name="svhn_cropped", split=tfds.Split.TEST)
    num_class = 10
    input_shape = (32,32,3)
    
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


    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    np.random.seed(SEED_NUM)
    idx = np.random.permutation(len(train_images))
    train_images = train_images[idx]
    train_labels = train_labels[idx]
    np.random.seed(SEED_NUM)
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
    test_dataset = []
    for i in range(num_class):
        train_dataset.append(images[np.where(labels == 1)[1]==i][:5400, :, :, :])
        val_dataset.append(images[np.where(labels == 1)[1]==i][5400:5600, :, :, :])
        test_dataset.append(images[np.where(labels == 1)[1]==i][5600:5800, :, :, :])
    
    return train_dataset, val_dataset, test_dataset, num_class, input_shape

# get train/validation/test dataset
def get_stl10_dataset():
    
    import resource
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

    train_dataset = tfds.load(name="stl10", split=tfds.Split.TRAIN)
    test_dataset = tfds.load(name="stl10", split=tfds.Split.TEST)
    unlabeled_dataset = tfds.load(name="stl10", split='unlabelled')

    num_class = 10
    input_shape = (96,96,3)
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    unlabeled_images = []
    
    for ex in train_dataset:
        train_images.append(ex['image'])
        train_labels.append(ex['label'])
    for ex in test_dataset:
        test_images.append(ex['image'])
        test_labels.append(ex['label'])
    for ex in unlabeled_dataset:
        unlabeled_images.append(ex['image'])
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    unlabeled_images = np.array(unlabeled_images)
    
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    unlabeled_images = unlabeled_images.astype('float32')
    
    train_images /=255.0
    test_images /= 255.0
    unlabeled_images /=255.0
    
    
    np.random.seed(SEED_NUM)
    idx = np.random.permutation(len(train_images))
    train_images = train_images[idx]
    train_labels = train_labels[idx]
    
    np.random.seed(SEED_NUM)
    idx = np.random.permutation(len(test_images))
    test_images = test_images[idx]
    test_labels = test_labels[idx]
    
    np.random.seed(SEED_NUM)
    idx = np.random.permutation(len(unlabeled_images))
    unlabeled_images = unlabeled_images[idx]
    
    
    images = np.concatenate((train_images, test_images), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis = 0)
    labels = tf.keras.utils.to_categorical(labels, num_class)
    
    train_labels = tf.keras.utils.to_categorical(train_labels, num_class)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_class)

    train_dataset = []
    val_dataset = []
    test_dataset = []
    for i in range(10):
        train_dataset.append(images[np.where(labels == 1)[1]==i][:1000, :, :, :])
        val_dataset.append(images[np.where(labels == 1)[1]==i][1000:1150, :, :, :])
        test_dataset.append(images[np.where(labels == 1)[1]==i][1150:1300, :, :, :])

    return train_dataset, val_dataset, test_dataset, unlabeled_images, num_class, input_shape    


# distribution for unlabeled data
def get_data_distribution(is_iid=True, is_xnid=False, num_dist_data=490, num_class=10):
    # num_dist_data
    #  iid: the number of labeled data per client
    #  non-iid: the number of unlabeled data per client
    if is_iid:
        ratio = [[num_dist_data//num_class for _ in range(num_class)] for _ in range(10)]
        #for i in ratio:
        #    for j in range(num_class):
        #        if i[j] == 0.5:
        #            i[j] = num_dist_data//num_class
        #return ratio
    else:
        if num_class == 10:
            if is_xnid:
                ratio = [
                [0.5,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.0,0.0], # type 0
                [0.0,0.5,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.0], # type 1 
                [0.0,0.0,0.5,0.1,0.1,0.1,0.05,0.05,0.05,0.05], # type 2 
                [0.05,0.0,0.0,0.5,0.1,0.1,0.1,0.05,0.05,0.05], # type 3 
                [0.05,0.05,0.0,0.0,0.5,0.1,0.1,0.1,0.05,0.05], # type 4 
                [0.05,0.05,0.05,0.0,0.0,0.5,0.1,0.1,0.1,0.05], # type 5 
                [0.05,0.05,0.05,0.05,0.0,0.0,0.5,0.1,0.1,0.1], # type 6 
                [0.1,0.05,0.05,0.05,0.05,0.0,0.0,0.5,0.1,0.1], # type 7 
                [0.1,0.1,0.05,0.05,0.05,0.05,0.0,0.0,0.5,0.1], # type 8 
                [0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.0,0.0,0.5], # type 9
            ]
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

            if num_dist_data == 490:
                dist_map = {0.5:243, 0.4:196, 0.2:98, 0.15:73, 0.1:49, 0.05:25, 0.03:15, 0.02:10, 0.0:0}
            elif num_dist_data == 540:
                dist_map = {0.5:249, 0.15:78, 0.1:54, 0.05:27, 0.03:20, 0.02:15, 0.0:0}
            elif num_dist_data == 50:
                dist_map = {0.5:23, 0.4:20, 0.2:10,  0.15:8, 0.1:5, 0.05:3, 0.0:0}
            else:
                dist_map = {r:int(num_dist_data*r) for r in ratio[0]}
            
            for ratio_type in ratio:
                for c in range(num_class):
                    ratio_type[c] = dist_map[ratio_type[c]]  

        elif num_class == 100:
            #Distribution per class, 5% * 10, 1.5% * 20, 0.3% * 40, 0.2% * 40
            base_dist = [0.05] * 10 + [0.015] * 20 + [0.003] * 60 + [0.002] * 10

            #Different distribution across clients
            ratio = []
            for i in range(10):
                curr_dist = copy.deepcopy(base_dist)
                curr_dist = curr_dist[i*10:] + curr_dist[:i*10]
                ratio.append(curr_dist)                

            #Proportion to actual number of data
            if num_dist_data == 4900:
                dist_map = {0.05:244, 0.015:73, 0.003:15, 0.002:10}            
            
            for ratio_type in ratio:
                for c in range(num_class):                    
                    ratio_type[c] = dist_map[ratio_type[c]]  

    print(ratio)
    
    return ratio


# distribute data for clients
def get_client_dataset(dataset, ratio_label, ratio_unlabel, num_client=100, num_label=5, num_class=10, seed=SEED_NUM):
    client_dataset = []

    for _ in range(num_client):
        temp = {}
        for i in range(num_class):
            temp[str(i)] = []
        temp['unlabel'] = []
        client_dataset.append(temp)

    client_list = list(range(num_client))
    random.seed(seed)
    random.shuffle(client_list)

    for label in range(num_class):
        num_data = len(dataset[label])
        idx = 0
        for client in client_list:        
            for _ in range(ratio_label[client%10][label]):
                client_dataset[client][str(label)].append(dataset[label][idx])
                idx += 1        

    random.shuffle(client_list)    
    
    # distribute unlabel
    for label in range(num_class):                
        idx = 0        
        for client in client_list:        
            for _ in range(ratio_unlabel[client%10][label]):
                client_dataset[client]['unlabel'].append(dataset[label][idx])
                idx += 1  
            
                        
    for i in range(num_client):
        for key in client_dataset[i]:
            client_dataset[i][key] = np.array(client_dataset[i][key])
            if i == (num_client-1):
                print("# of labeled data (class {}): {}".format(key, len(client_dataset[i][key])))
        client_dataset[i]['unlabel'] = np.array(client_dataset[i]['unlabel'])
        if i == num_client-1:
            print("# of unlabeled data: {}".format(len(client_dataset[i]['unlabel'])))

    return client_dataset



# distribute data for clients for supervised learnings
def get_client_dataset_sl(dataset, ratio, num_client=100, num_label=54, num_class=10):            
    client_dataset = []
    client_labels = []

    for _ in range(num_client):
        temp = {}
        temp2 = {}
        for i in range(num_class):
            temp[str(i)] = []        
            temp2[str(i)] = []        
        client_dataset.append(temp)
        client_labels.append(temp2)

    # Distribute the data according to ratio
    for label in range(num_class):        
        idx = 0
        # distribute label
        for client in range(num_client):
            for _ in range(ratio[client%10][label]):                
                client_dataset[client][str(label)].append(dataset[label][idx])
                client_labels[client][str(label)].append(label)
                idx += 1                
        
    # Make the data numpy array
    for i in range(num_client):
        for key in client_dataset[i]:
            client_dataset[i][key] = np.array(client_dataset[i][key])
            client_labels[i][key] = np.array(client_labels[i][key])
            if i == (num_client-1):
                print("# of labeled data, in client 0 (class {}): {}".format(key, len(client_dataset[i][key])))
    
    return client_dataset, client_labels



# distribute data for clients
def get_client_dataset_stl10(dataset, unlabeled_dataset, num_client=100, num_label=10, num_class=10, num_unlabel=980):
    client_dataset = []

    for _ in range(num_client):
        temp = {}
        for i in range(num_class):
            temp[str(i)] = []
        temp['unlabel'] = []
        client_dataset.append(temp)

    for label in range(num_class):
        num_data = len(dataset[label])
        idx = 0
        # distribute label
        for client in range(num_client):
            for _ in range(num_label):
                client_dataset[client][str(label)].append(dataset[label][idx])
                idx += 1
                
        # distribute unlabel
        for client in range(num_client):
            client_dataset[client]['unlabel'] = unlabeled_dataset[client*num_unlabel:(client+1)*num_unlabel, :, :, :]
                              
    for i in range(num_client):
        for key in client_dataset[i]:
            client_dataset[i][key] = np.array(client_dataset[i][key])
            if i == (num_client-1):
                print("# of labeled data (class {}): {}".format(key, len(client_dataset[i][key])))
        client_dataset[i]['unlabel'] = np.array(client_dataset[i]['unlabel'])
        if i == num_client-1:
            print("# of unlabeled data: {}".format(len(client_dataset[i]['unlabel'])))

    
    return client_dataset


def get_dataset(dataset_name='cifar10', is_iid=True, is_xnid=False, num_client=100, num_label=5, num_unlabel=490, is_sl=False):
    if dataset_name == 'cifar10':
        train_dataset, val_dataset, test_dataset, num_class, input_shape = get_cifar10_dataset()
    elif dataset_name == 'cifar100':
        train_dataset, val_dataset, test_dataset, num_class, input_shape = get_cifar100_dataset()
    elif dataset_name == 'svhn':
        train_dataset, val_dataset, test_dataset, num_class, input_shape = get_svhn_dataset()
    elif dataset_name == 'stl10':
        train_dataset, val_dataset, test_dataset, unlabeled_images, num_class, input_shape = get_stl10_dataset()    
    else: 
        print("Invalid dataset name")
        return

    # Prepare dataset for supervised learning
    if is_sl:
        ratio = get_data_distribution(is_iid=is_iid, is_xnid=is_xnid, num_dist_data=num_label * num_class, num_class=num_class)           
        client_dataset, client_labels = get_client_dataset_sl(train_dataset, ratio, num_client=num_client, num_label=num_label, num_class=num_class)
    # Prepare dataset for ProtoFSSL
    elif dataset_name in ['cifar10', 'svhn', 'cifar100']:
        if is_xnid:
            ratio_label = get_data_distribution(is_iid=is_iid, is_xnid=is_xnid, num_dist_data=num_label * num_class, num_class=num_class)
        else:
            ratio_label = get_data_distribution(is_iid=True, is_xnid=False, num_dist_data=num_label * num_class, num_class=num_class)
        ratio_unlabel = get_data_distribution(is_iid=is_iid, is_xnid=is_xnid, num_dist_data=num_unlabel, num_class=num_class)        
        client_dataset = get_client_dataset(train_dataset, ratio_label, ratio_unlabel, num_client=num_client, num_label=num_label, num_class=num_class)
        client_labels = None        
    else:
        client_dataset = get_client_dataset_stl10(train_dataset, unlabeled_images, num_client=num_client, num_label=num_label, num_class=num_class)
        client_labels = None

    return client_dataset, val_dataset, test_dataset, num_class, input_shape, client_labels



    