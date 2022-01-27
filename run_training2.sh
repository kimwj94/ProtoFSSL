#!/bin/bash
# FedAvg -Full SL
#python proto_fssl.py --exp_name fedavg_cifar10 --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2
#python proto_fssl.py --exp_name fedavg_cifar10_nid --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid

#base models, iid
#python proto_fssl.py --exp_name cifar10 --dataset cifar10 --model res9 
#python proto_fssl.py --exp_name test_svhn --dataset svhn --model res9 
#python proto_fssl.py --exp_name test_stl10 --dataset stl10 --model res9 --num_label 10 --num_unlabel 980

# for i in {1..3}
# do
#     python proto_fssl.py --exp_name stl10 --dataset stl10 --model res9 --num_label 10 --num_unlabel 980 --bn_type bn
# done

#With BN
#for i in {1..3}
#do
#    python proto_fssl.py --exp_name svhn_bn --dataset svhn --model res9 --bn_type bn
#    python proto_fssl.py --exp_name svhn_bn_nid --dataset svhn --model res9 --bn_type bn --non_iid
#done

#python proto_fssl.py --exp_name fedprox_stl10_1 --dataset stl10 --model res9 --is_sl --num_label 10 --local_episode 10 --mu 1e-2 --fl_framework fedprox
#python proto_fssl.py --exp_name fedprox_stl10_2 --dataset stl10 --model res9 --is_sl --num_label 10 --local_episode 10 --mu 1e-3 --fl_framework fedprox
#python proto_fssl.py --exp_name fedprox_stl10_3 --dataset stl10 --model res9 --is_sl --num_label 10 --local_episode 10 --mu 1e-4 --fl_framework fedprox

python proto_fssl.py --exp_name fedprox_1 --dataset cifar10 --model res9 --is_sl --num_label 5 --local_episode 2 --mu 1e-2 --fl_framework fedprox
python proto_fssl.py --exp_name fedprox_2 --dataset cifar10 --model res9 --is_sl --num_label 5 --local_episode 2 --mu 1e-3 --fl_framework fedprox
python proto_fssl.py --exp_name fedprox_3 --dataset cifar10 --model res9 --is_sl --num_label 5 --local_episode 2 --mu 1e-4 --fl_framework fedprox
python proto_fssl.py --exp_name fedprox_4 --dataset cifar10 --model res9 --is_sl --num_label 5 --local_episode 2 --mu 1e-1 --fl_framework fedprox

python proto_fssl.py --exp_name fedprox_svhn_1 --dataset svhn --model res9 --is_sl --num_label 5 --local_episode 2 --mu 1e-2 --fl_framework fedprox
python proto_fssl.py --exp_name fedprox_svhn_2 --dataset svhn --model res9 --is_sl --num_label 5 --local_episode 2 --mu 1e-3 --fl_framework fedprox
python proto_fssl.py --exp_name fedprox_svhn_3 --dataset svhn --model res9 --is_sl --num_label 5 --local_episode 2 --mu 1e-4 --fl_framework fedprox
python proto_fssl.py --exp_name fedprox_svhn_4 --dataset svhn --model res9 --is_sl --num_label 5 --local_episode 2 --mu 1e-1 --fl_framework fedprox

#Ablations upper bound
# for i in {1..3}
# do
#     python proto_fssl.py --exp_name fedavg_cifar10_bn --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --bn_type bn
#     python proto_fssl.py --exp_name fedavg_cifar10_bn_nid --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid --bn_type bn
#     python proto_fssl.py --exp_name fedavg_cifar10_gn --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --bn_type gn
#     python proto_fssl.py --exp_name fedavg_cifar10_gn_nid --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid --bn_type gn
#     python proto_fssl.py --exp_name fedavg_cifar10_sbn --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --bn_type sbn
#     python proto_fssl.py --exp_name fedavg_cifar10_sbn_nid --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid --bn_type sbn
#     python proto_fssl.py --exp_name fedavg_cifar10_res18_sbn --dataset cifar10 --model res18 --is_sl --num_label 54  --local_episode 2 --bn_type sbn
#     python proto_fssl.py --exp_name fedavg_cifar10_res18_sbn_nid --dataset cifar10 --model res18 --is_sl --num_label 54  --local_episode 2 --non_iid --bn_type sbn
# done

#Ablations proposal
# for i in {1..3}
# do
#     python proto_fssl.py --exp_name pfssl_cifar10_bn --dataset cifar10 --model res9 --bn_type bn
#     python proto_fssl.py --exp_name pfssl_cifar10_bn_nid --dataset cifar10 --model res9 --non_iid --bn_type bn
#     python proto_fssl.py --exp_name pfssl_cifar10_gn --dataset cifar10 --model res9 --bn_type gn
#     python proto_fssl.py --exp_name pfssl_cifar10_gn_nid --dataset cifar10 --model res9 --non_iid --bn_type gn
#     python proto_fssl.py --exp_name pfssl_cifar10_sbn --dataset cifar10 --model res9  --bn_type sbn
#     python proto_fssl.py --exp_name pfssl_cifar10_sbn_nid --dataset cifar10 --model res9 --non_iid --bn_type sbn
#     python proto_fssl.py --exp_name pfssl_cifar10_res18_sbn --dataset cifar10 --model res18 --bn_type sbn
#     python proto_fssl.py --exp_name pfssl_cifar10_res18_sbn_nid --dataset cifar10 --model res18 --non_iid --bn_type sbn
# done




#python proto_fssl.py --exp_name proposal_res18_bn10 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9
#python proto_fssl.py --exp_name proposal_res18_bn11 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-3 --opt_momentum 0.9
#python proto_fssl.py --exp_name proposal_res18_bn12 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-3 
#python proto_fssl.py --exp_name proposal_res18_bn13 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9 --keep_proto_rounds 5 --refine_at_begin
#python proto_fssl.py --exp_name proposal_res18_bn14 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9 --keep_proto_rounds 5 --refine_at_begin --warmup_episode 3
#python proto_fssl.py --exp_name proposal_res18_bn15 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9 --l2_factor 1e-3

# With augmentation on sharing prototypes
#python proto_fssl.py --exp_name proposal_res18_bn16 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9

#With sBN
#python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --bn_type sbn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9
#python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --bn_type sbn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9 --non_iid



#base models, non-iid
#python proto_fssl.py --exp_name test_cifar10_nid --dataset cifar10 --model res9 --num_round 5 --non_iid
#python proto_fssl.py --exp_name test_svhn_nid --dataset svhn --model res9 --num_round 5

######## Ablations
#python proto_fssl.py --exp_name proposal_bn --dataset cifar10 --model res9 --bn_type bn --num_round 5
#for i in {1..3}
#do
#    python proto_fssl.py --exp_name proposal_gn --dataset cifar10 --model res9 --bn_type gn
#done
#for i in {1..3}
#do
#    python proto_fssl.py --exp_name test_refine_begin_keep5 --dataset cifar10 --model res9 --keep_proto_rounds 5 --refine_at_begin
#done
#python proto_fssl.py --exp_name proposal_sbn --dataset cifar10 --model res9 --bn_type sbn --num_round 5
#python proto_fssl.py --exp_name proposal_res18_sbn --dataset cifar10 --model res18 --bn_type sbn --num_round 5
#python proto_fssl.py --exp_name proposal_wres28x2_sbn --dataset cifar10 --model wres28x2 --bn_type sbn --num_round 5

#python proto_fssl.py --exp_name proposal_res18_bn1 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer sgd --opt_lr 0.1 --opt_momentum 0.7
#python proto_fssl.py --exp_name proposal_res18_bn2 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9
#python proto_fssl.py --exp_name proposal_res18_bn3 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer sgd --opt_lr 0.01 --opt_momentum 0.9
#With learning schedule
#python proto_fssl.py --exp_name proposal_res18_bn4 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9
#With bigger l2 regularization
#python proto_fssl.py --exp_name proposal_res18_bn5 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9
#With augmentation
#python proto_fssl.py --exp_name proposal_res18_bn6 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9
#python proto_fssl.py --exp_name proposal_res18_bn7 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4 --opt_momentum 0.9

#Change resnet18 function
#python proto_fssl.py --exp_name proposal_res18_bn8 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-3 --opt_momentum 0.9
#python proto_fssl.py --exp_name proposal_res18_bn9 --dataset cifar10 --model res18 --bn_type bn --num_round 300 --optimizer rmsprop --opt_lr 1e-4

#python proto_fssl.py --exp_name proposal_bn_nid --dataset cifar10 --model res9 --bn_type bn --non_iid --num_round 5
#python proto_fssl.py --exp_name proposal_gn_nid --dataset cifar10 --model res9 --bn_type gn --non_iid --num_round 5
#python proto_fssl.py --exp_name proposal_sbn_nid --dataset cifar10 --model res9 --bn_type sbn --non_iid --num_round 5
#python proto_fssl.py --exp_name proposal_res18_sbn_nid --dataset cifar10 --model res18 --bn_type sbn --non_iid --num_round 5
#python proto_fssl.py --exp_name proposal_wres28x2_sbn_nid --dataset cifar10 --model wres28x2 --bn_type sbn --non_iid --num_round 5