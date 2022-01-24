#!/bin/bash

#base models, iid
python proto_fssl.py --exp_name test_cifar10 --dataset cifar10 --model res9 --num_round 5
python proto_fssl.py --exp_name test_svhn --dataset svhn --model res9 --num_round 5
python proto_fssl.py --exp_name test_stl10 --dataset stl10 --model res9 --num_round 5 --num_label 10 --num_unlabel 980

#base models, non-iid
python proto_fssl.py --exp_name test_cifar10_nid --dataset cifar10 --model res9 --num_round 5 --non_iid
python proto_fssl.py --exp_name test_svhn_nid --dataset svhn --model res9 --num_round 5

######## Ablations
python proto_fssl.py --exp_name proposal_bn --dataset cifar10 --model res9 --bn_type bn --num_round 5
python proto_fssl.py --exp_name proposal_gn --dataset cifar10 --model res9 --bn_type gn --num_round 5
python proto_fssl.py --exp_name proposal_sbn --dataset cifar10 --model res9 --bn_type sbn --num_round 5
python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --bn_type sbn --num_round 5
python proto_fssl.py --exp_name proposal_wres28x2_sbn --dataset cifar10 --model wres28x2 --bn_type sbn --num_round 5

python proto_fssl.py --exp_name proposal_bn_nid --dataset cifar10 --model res9 --bn_type bn --non_iid --num_round 5
python proto_fssl.py --exp_name proposal_gn_nid --dataset cifar10 --model res9 --bn_type gn --non_iid --num_round 5
python proto_fssl.py --exp_name proposal_sbn_nid --dataset cifar10 --model res9 --bn_type sbn --non_iid --num_round 5
python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --bn_type sbn --non_iid --num_round 5
python proto_fssl.py --exp_name proposal_wres28x2_sbn_nid --dataset cifar10 --model wres28x2 --bn_type sbn --non_iid --num_round 5


#non-iid bn
#python proto_fssl.py --exp_name proposal_bn_nid --dataset cifar10 --model res9 --non_iid --bn_type bn
#python proto_fssl.py --exp_name proposal_bn_nid --dataset cifar10 --model res9 --non_iid --bn_type bn
#python proto_fssl.py --exp_name proposal_bn_nid --dataset cifar10 --model res9 --non_iid --bn_type bn

#non-iid gn
#python proto_fssl.py --exp_name proposal_gn_nid --dataset cifar10 --model res9 --non_iid --bn_type gn
#python proto_fssl.py --exp_name proposal_gn_nid --dataset cifar10 --model res9 --non_iid --bn_type gn
#python proto_fssl.py --exp_name proposal_gn_nid --dataset cifar10 --model res9 --non_iid --bn_type gn

#non-iid sbn
#python proto_fssl.py --exp_name proposal_sbn_nid --dataset cifar10 --model res9 --non_iid --bn_type sbn
#python proto_fssl.py --exp_name proposal_sbn_nid --dataset cifar10 --model res9 --non_iid --bn_type sbn
#python proto_fssl.py --exp_name proposal_sbn_nid --dataset cifar10 --model res9 --non_iid --bn_type sbn

#sbn resnet18
#python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --non_iid --bn_type sbn
#python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --non_iid --bn_type sbn
#python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --non_iid --bn_type sbn

#non-iid sbn resnet18
#python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --non_iid --bn_type sbn
#python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --non_iid --bn_type sbn
#python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --non_iid --bn_type sbn