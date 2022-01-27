#!/bin/bash
#FedProx parameter tuning
#python proto_fssl.py --exp_name fedprox_nid_1 --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-2 --fl_framework fedprox --non_iid
#python proto_fssl.py --exp_name fedprox_nid_2 --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-3 --fl_framework fedprox --non_iid
#python proto_fssl.py --exp_name fedprox_nid_3 --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-4 --fl_framework fedprox --non_iid

#python proto_fssl.py --exp_name fedprox_svhn_1 --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-2 --fl_framework fedprox
#python proto_fssl.py --exp_name fedprox_svhn_2 --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-3 --fl_framework fedprox
#python proto_fssl.py --exp_name fedprox_svhn_3 --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-4 --fl_framework fedprox

#python proto_fssl.py --exp_name fedprox_svhn_nid_1 --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-2 --fl_framework fedprox --non_iid
#python proto_fssl.py --exp_name fedprox_svhn_nid_2 --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-3 --fl_framework fedprox --non_iid
#python proto_fssl.py --exp_name fedprox_svhn_nid_3 --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-4 --fl_framework fedprox --non_iid

#python proto_fssl.py --exp_name fedprox_part_1 --dataset cifar10 --model res9 --is_sl --num_label 5  --local_episode 2 --mu 1e-2 --fl_framework fedprox
#python proto_fssl.py --exp_name fedprox_part_2 --dataset cifar10 --model res9 --is_sl --num_label 5  --local_episode 2 --mu 1e-3 --fl_framework fedprox
#python proto_fssl.py --exp_name fedprox_part_3 --dataset cifar10 --model res9 --is_sl --num_label 5  --local_episode 2 --mu 1e-4 --fl_framework fedprox

#python proto_fssl.py --exp_name fedprox_svhn_part_1 --dataset svhn --model res9 --is_sl --num_label 5  --local_episode 2 --mu 1e-2 --fl_framework fedprox
#python proto_fssl.py --exp_name fedprox_svhn_part_2 --dataset svhn --model res9 --is_sl --num_label 5  --local_episode 2 --mu 1e-3 --fl_framework fedprox
#python proto_fssl.py --exp_name fedprox_svhn_part_3 --dataset svhn --model res9 --is_sl --num_label 5  --local_episode 2 --mu 1e-4 --fl_framework fedprox


#python proto_fssl.py --exp_name fedavg_part --dataset cifar10 --model res9 --is_sl --num_label 5  --local_episode 2
python proto_fssl.py --exp_name pfssl_fedprox_cifar10 --dataset cifar10 --model res9 --num_label 5  --mu 1e-1 --fl_framework fedprox
python proto_fssl.py --exp_name pfssl_fedprox_cifar10 --dataset cifar10 --model res9 --num_label 5  --mu 1e-2 --fl_framework fedprox
python proto_fssl.py --exp_name pfssl_fedprox_cifar10 --dataset cifar10 --model res9 --num_label 5  --mu 1e-3 --fl_framework fedprox
python proto_fssl.py --exp_name pfssl_fedprox_cifar10 --dataset cifar10 --model res9 --num_label 5  --mu 1e-4 --fl_framework fedprox

 
python proto_fssl.py --exp_name pfssl_fedprox_cifar10_nid --dataset cifar10 --model res9 --num_label 5  --mu 1e-1 --fl_framework fedprox --non_iid
python proto_fssl.py --exp_name pfssl_fedprox_cifar10_nid --dataset cifar10 --model res9 --num_label 5  --mu 1e-2 --fl_framework fedprox --non_iid
python proto_fssl.py --exp_name pfssl_fedprox_cifar10_nid --dataset cifar10 --model res9 --num_label 5  --mu 1e-3 --fl_framework fedprox --non_iid
python proto_fssl.py --exp_name pfssl_fedprox_cifar10_nid --dataset cifar10 --model res9 --num_label 5  --mu 1e-4 --fl_framework fedprox --non_iid


python proto_fssl.py --exp_name pfssl_fedprox_svhn --dataset svhn --model res9 --num_label 5  --mu 1e-1 --fl_framework fedprox
python proto_fssl.py --exp_name pfssl_fedprox_svhn --dataset svhn --model res9 --num_label 5  --mu 1e-2 --fl_framework fedprox
python proto_fssl.py --exp_name pfssl_fedprox_svhn --dataset svhn --model res9 --num_label 5  --mu 1e-3 --fl_framework fedprox
python proto_fssl.py --exp_name pfssl_fedprox_svhn --dataset svhn --model res9 --num_label 5  --mu 1e-4 --fl_framework fedprox

 
python proto_fssl.py --exp_name pfssl_fedprox_svhn_nid --dataset svhn --model res9 --num_label 5  --mu 1e-1 --fl_framework fedprox --non_iid
python proto_fssl.py --exp_name pfssl_fedprox_svhn_nid --dataset svhn --model res9 --num_label 5  --mu 1e-2 --fl_framework fedprox --non_iid
python proto_fssl.py --exp_name pfssl_fedprox_svhn_nid --dataset svhn --model res9 --num_label 5  --mu 1e-3 --fl_framework fedprox --non_iid
python proto_fssl.py --exp_name pfssl_fedprox_svhn_nid --dataset svhn --model res9 --num_label 5  --mu 1e-4 --fl_framework fedprox --non_iid
