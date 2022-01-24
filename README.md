# ProtoFSSL: Federated Semi-Supervised Learning with Prototypical Networks

### Environment
To install required package, create an anaconda environment with below command.
```
conda env create -f environment_protofssl.yaml
```
You can start training by 
```
python proto_fssl.py --exp_name test_cifar10 --dataset cifar10 --model res9 --num_round 5
```

or use bash file
```
bash run_training.sh
```

The results is saved in result folder.

---
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
