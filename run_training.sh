#!/bin/bash

#base models, iid
#python proto_fssl.py --exp_name test_normalize --dataset cifar10 --model res9 --num_round 5
#python proto_fssl.py --exp_name test_normalize_keep5 --dataset cifar10 --model res9 --keep_proto_rounds 5
#python proto_fssl.py --exp_name test_svhn --dataset svhn --model res9 --num_round 5
#python proto_fssl.py --exp_name test_stl10 --dataset stl10 --model res9 --num_round 5 --num_label 10 --num_unlabel 980

#base models, non-iid
#python proto_fssl.py --exp_name test_cifar10_nid --dataset cifar10 --model res9 --num_round 5 --non_iid
#python proto_fssl.py --exp_name test_svhn_nid --dataset svhn --model res9 --num_round 5

######## Ablations
#python proto_fssl.py --exp_name proposal_bn --dataset cifar10 --model res9 --bn_type bn --num_round 5
for i in {1..3}
do
    python proto_fssl.py --exp_name proposal_gn --dataset cifar10 --model res9 --bn_type gn
done
#python proto_fssl.py --exp_name proposal_sbn --dataset cifar10 --model res9 --bn_type sbn --num_round 5
#python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --bn_type sbn --num_round 5
#python proto_fssl.py --exp_name proposal_wres28x2_sbn --dataset cifar10 --model wres28x2 --bn_type sbn --num_round 5

#python proto_fssl.py --exp_name proposal_res18_bn1 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer sgd --opt_lr 0.1 --opt_momentum 0.7
#python proto_fssl.py --exp_name proposal_res18_bn2 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9
#python proto_fssl.py --exp_name proposal_res18_bn3 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer sgd --opt_lr 0.01 --opt_momentum 0.9
#With learning schedule
#python proto_fssl.py --exp_name proposal_res18_bn4 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9

#python proto_fssl.py --exp_name proposal_bn_nid --dataset cifar10 --model res9 --bn_type bn --non_iid --num_round 5
#python proto_fssl.py --exp_name proposal_gn_nid --dataset cifar10 --model res9 --bn_type gn --non_iid --num_round 5
#python proto_fssl.py --exp_name proposal_sbn_nid --dataset cifar10 --model res9 --bn_type sbn --non_iid --num_round 5
#python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --bn_type sbn --non_iid --num_round 5
#python proto_fssl.py --exp_name proposal_wres28x2_sbn_nid --dataset cifar10 --model wres28x2 --bn_type sbn --non_iid --num_round 5

# #non-iid bn
# for i in {1..3}
# do    
#     python proto_fssl.py --exp_name proposal_bn_nid --dataset cifar10 --model res9 --non_iid --bn_type bn
# done

# #non-iid gn
# for i in {1..3}
# do    
#     python proto_fssl.py --exp_name proposal_gn_nid --dataset cifar10 --model res9 --non_iid --bn_type gn
# done

# #non-iid sbn
# for i in {1..3}
# do    
#     python proto_fssl.py --exp_name proposal_sbn_nid --dataset cifar10 --model res9 --non_iid --bn_type sbn
# done

# #non-iid resnet18 sbn 
# for i in {1..3}
# do    
#     python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --non_iid --bn_type sbn
# done

# #non-iid wres28x2 sbn 
# for i in {1..3}
# do    
#     python proto_fssl.py --exp_name proposal_wres28x2_sbn_nid --dataset cifar10 --model wres28x2 --non_iid --bn_type sbn
# done

# #resnet18 sbn 
# for i in {1..3}
# do    
#     python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --bn_type sbn
# done

# #wres28x2 sbn 
# for i in {1..3}
# do    
#     python proto_fssl.py --exp_name proposal_wres28x2_sbn --dataset cifar10 --model wres28x2 --bn_type sbn
# done