#!/bin/bash

#base model
python proto_fssl.py --exp_name proposal_base --dataset cifar10 --model res9 
python proto_fssl.py --exp_name proposal_base --dataset cifar10 --model res9 
python proto_fssl.py --exp_name proposal_base --dataset cifar10 --model res9 

#sbn resnet18
python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --non_iid --bn_type sbn
python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --non_iid --bn_type sbn
python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --non_iid --bn_type sbn

#non-iid bn
python proto_fssl.py --exp_name proposal_bn_nid --dataset cifar10 --model res9 --non_iid --bn_type bn
python proto_fssl.py --exp_name proposal_bn_nid --dataset cifar10 --model res9 --non_iid --bn_type bn
python proto_fssl.py --exp_name proposal_bn_nid --dataset cifar10 --model res9 --non_iid --bn_type bn

#non-iid gn
python proto_fssl.py --exp_name proposal_gn_nid --dataset cifar10 --model res9 --non_iid --bn_type gn
python proto_fssl.py --exp_name proposal_gn_nid --dataset cifar10 --model res9 --non_iid --bn_type gn
python proto_fssl.py --exp_name proposal_gn_nid --dataset cifar10 --model res9 --non_iid --bn_type gn

#non-iid sbn
python proto_fssl.py --exp_name proposal_sbn_nid --dataset cifar10 --model res9 --non_iid --bn_type sbn
python proto_fssl.py --exp_name proposal_sbn_nid --dataset cifar10 --model res9 --non_iid --bn_type sbn
python proto_fssl.py --exp_name proposal_sbn_nid --dataset cifar10 --model res9 --non_iid --bn_type sbn

#non-iid sbn resnet18
python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --non_iid --bn_type sbn
python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --non_iid --bn_type sbn
python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --non_iid --bn_type sbn