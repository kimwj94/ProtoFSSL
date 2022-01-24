import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import tensorflow_datasets as tfds

SEED_NUM = 1001
# get train/validation/test dataset
def get_cifar10_dataset():
    
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

    images /= 255.0

    train_dataset = []
    val_dataset = []
    test_dataset = []
    for i in range(num_class):
        train_dataset.append(images[np.where(labels == 1)[1]==i][:5400, :, :, :])
        val_dataset.append(images[np.where(labels == 1)[1]==i][5400:5700, :, :, :])
        test_dataset.append(images[np.where(labels == 1)[1]==i][5700:, :, :, :])
    
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
    
    #train_labels = tf.keras.utils.to_categorical(train_labels, num_class)
    #test_labels = tf.keras.utils.to_categorical(test_labels, num_class)

    train_dataset = []
    val_dataset = []
    test_dataset = []
    for i in range(10):
        train_dataset.append(images[np.where(labels == 1)[1]==i][:1000, :, :, :])
        val_dataset.append(images[np.where(labels == 1)[1]==i][1000:1150, :, :, :])
        test_dataset.append(images[np.where(labels == 1)[1]==i][1150:1300, :, :, :])

    return train_dataset, val_dataset, test_dataset, unlabeled_images, num_class, input_shape


# distribution for unlabeled data
def get_data_distribution(is_iid=True, num_unlabel=490):
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


# distribute data for clients
def get_client_dataset(dataset, ratio, num_client=100, num_label=5, num_class=10):            
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
            for _ in range(ratio[client%10][label]):
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


def get_dataset(dataset_name='cifar10', is_iid=True, num_client=100, num_label=5, num_unlabel=490):
    if dataset_name == 'cifar10':
        train_dataset, val_dataset, test_dataset, num_class, input_shape = get_cifar10_dataset()
    elif dataset_name == 'svhn':
        train_dataset, val_dataset, test_dataset, num_class, input_shape = get_svhn_dataset()
    elif dataset_name == 'stl10':
        train_dataset, val_dataset, test_dataset, unlabeled_images, num_class, input_shape = get_stl10_dataset()    
    else: 
        print("Invalid dataset name")
        return

    if dataset_name == 'cifar10' or dataset_name == 'svhn':
        ratio = get_data_distribution(is_iid=is_iid, num_unlabel=num_unlabel)
        client_dataset = get_client_dataset(train_dataset, ratio, num_client=num_client, num_label=num_label, num_class=num_class)
    else:
        client_dataset = get_client_dataset_stl10(train_dataset, unlabeled_images, num_client=num_client, num_label=num_label, num_class=num_class)

    return client_dataset, val_dataset, test_dataset, num_class, input_shape



    