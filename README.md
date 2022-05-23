# ProtoFSSL: Federated Semi-Supervised Learning with Prototype-based Consistency Regularization
This repository is the official implementation of *ProtoFSSL: Federated Semi-Supervised Learning with Prototype-based Consistency Regularization*.

**Overall design of *ProtoFSSL*.**
![design](./img/design.jpg)



## Requirements

To install requirements:

```setup
conda env create -f environment_protofssl.yaml
```


## Training

To train the model(s) in the paper, run this command:

```
python proto_fssl.py --exp_name test_cifar10 --dataset cifar10 --model res9 --num_round 5
```
If you want FedProx framework, (default is FedAvg)
```
python proto_fssl.py --exp_name test_cifar10 --dataset cifar10 --model res9 --num_round 5 --fl_framework fedprox --mu 1e-3
```


The results is saved in result folder.


### Summary of input arguments
```
exp_name: Experiment name
dataset: The name of the datset. One of [cifar10, svhn, stl10], default: cifar10
model: Model type. One of [res9, res18, wres28x2]
bn_type: Batch normalization type one of [bn, sbn, gn], default: None
non_iid: Run non-iid distributed data
num_round: Number of training round, default: 300
num_label: Number of labeled data per client per class, default: 5
num_unlabel: Number of unlabeled data per client, default: 490
local_episode: Number of local episode, default: 10
unlabel_round: Starting training round to use unlabeled data(non-inclusive), default: 0
optimizer: Which optimizer to use(rmsprop, sgd, adam), default: rmsprop
num_client: Number of clients, default: 100
num_active_client: Number of active clients, default: 5
unlabel_loss_type: Loss type to train unlabeled data. one of [CE, MSE], default: CE
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
