#!/bin/bash
# FedAvg -Full SL
# python proto_fssl.py --exp_name fedavg_cifar10 --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2
# python proto_fssl.py --exp_name fedavg_cifar10_nid --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid
# python proto_fssl.py --exp_name fedavg_svhn --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2
# python proto_fssl.py --exp_name fedavg_svhn_nid --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid

# # FedProx -Full SL
# python proto_fssl.py --exp_name fedprox_cifar10 --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-2 --fl_framework fedprox
# python proto_fssl.py --exp_name fedprox_cifar10_nid --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-2 --fl_framework fedprox --non_iid
# python proto_fssl.py --exp_name fedprox_svhn --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-4 --fl_framework fedprox
# python proto_fssl.py --exp_name fedprox_svhn_nid --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2 --mu 1e-4 --fl_framework fedprox --non_iid

# #FedAvg - Partial SL
# python proto_fssl.py --exp_name fedavg_part_cifar10 --dataset cifar10 --model res9 --is_sl --num_label 5 --num_unlabel 0 --local_episode 2
# python proto_fssl.py --exp_name fedavg_part_cifar10_nid --dataset cifar10 --model res9 --is_sl --num_label 5 --num_unlabel 0 --local_episode 2 --non_iid
# python proto_fssl.py --exp_name fedavg_part_svhn --dataset svhn --model res9 --is_sl --num_label 5 --num_unlabel 0 --local_episode 2
# python proto_fssl.py --exp_name fedavg_part_svhn_nid --dataset svhn --model res9 --is_sl --num_label 5 --num_unlabel 0 --local_episode 2 --non_iid
# python proto_fssl.py --exp_name fedavg_part_stl10 --dataset stl10 --model res9 --num_label 10 --num_unlabel 0 --local_episode 2

# #FedProx - Partial SL
# python proto_fssl.py --exp_name fedprox_part_cifar10 --dataset cifar10 --model res9 --is_sl --num_label 5 --num_unlabel 0 --local_episode 2 --mu 1e-2 --fl_framework fedprox
# python proto_fssl.py --exp_name fedprox_part_cifar10_nid --dataset cifar10 --model res9 --is_sl --num_label 5 --num_unlabel 0 --local_episode 2 --mu 1e-2 --fl_framework fedprox --non_iid
# python proto_fssl.py --exp_name fedprox_part_svhn --dataset svhn --model res9 --is_sl --num_label 5 --num_unlabel 0 --local_episode 2 --mu 1e-4 --fl_framework fedprox
# python proto_fssl.py --exp_name fedprox_part_svhn_nid --dataset svhn --model res9 --is_sl --num_label 5 --num_unlabel 0 --local_episode 2 --mu 1e-4 --fl_framework fedprox --non_iid
# python proto_fssl.py --exp_name fedprox_part_stl10 --dataset stl10 --model res9 --is_sl --num_label 10 --num_unlabel 0 --local_episode 2 --mu 1e-2 --fl_framework fedprox

#base models, iid
python proto_fssl.py --exp_name cifar10 --dataset cifar10 --model res9 
python proto_fssl.py --exp_name svhn --dataset svhn --model res9 
python proto_fssl.py --exp_name stl10 --dataset stl10 --model res9 --num_label 10 --num_unlabel 980

#base models, non-iid
python proto_fssl.py --exp_name cifar10_nid --dataset cifar10 --model res9 --non_iid
python proto_fssl.py --exp_name svhn_nid --dataset svhn --model res9 --non_iid


#base models, severe non-iid
python proto_fssl.py --exp_name fedavg_xnid --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid --extreme_non_iid --seed 101 
python proto_fssl.py --exp_name fedavg_part_xnid --model res9 --is_sl --num_label 5  --local_episode 2 --non_iid --extreme_non_iid --seed 101 
python proto_fssl.py --exp_name fixmatch_xnid --model res9  --fixmatch --local_episode 2 --non_iid --extreme_non_iid --seed 101 
python proto_fssl.py --exp_name cifar10_xnid --dataset cifar10 --model res9 --non_iid --extreme_non_iid --local_episode 15 --q_unlabel 200 --unlabel_round 200 --keep_proto_rounds 10 --helper_cnt 15 --seed 101

python proto_fssl.py --exp_name fedavg_svhn_xnid --dataset svhn --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid --extreme_non_iid --seed 101 
python proto_fssl.py --exp_name fedavg_part_xnid --dataset svhn --model res9 --is_sl --num_label 5  --local_episode 2 --non_iid --extreme_non_iid --seed 101 
python proto_fssl.py --exp_name fixmatch_svhn_xnid --dataset svhn --model res9  --fixmatch --local_episode 2 --non_iid --extreme_non_iid --seed 101 
python proto_fssl.py --exp_name svhn_xnid --dataset svhn --model res9 --non_iid --extreme_non_iid --local_episode 15 --q_unlabel 200 --unlabel_round 200 --keep_proto_rounds 10 --helper_cnt 15 --seed 101 


#fixmatch, severe non-iid
# python proto_fssl.py --exp_name cifar10_fixmatch --dataset cifar10 --model res9 --fixmatch --seed 1001 --local_episode 2 --non_iid

#With BN
python proto_fssl.py --exp_name cifar10_bn --dataset cifar10 --model res9 --bn_type bn
python proto_fssl.py --exp_name cifar10_bn_nid --dataset cifar10 --model res9 --bn_type bn --non_iid
python proto_fssl.py --exp_name svhn_bn --dataset svhn --model res9 --bn_type bn
python proto_fssl.py --exp_name svhn_bn_nid --dataset svhn --model res9 --bn_type bn --non_iid
python proto_fssl.py --exp_name stl10_bn --dataset stl10 --model res9 --num_label 10 --num_unlabel 980 --bn_type bn

python proto_fssl.py --exp_name fedprox_cifar10_bn --dataset cifar10 --model res9 --num_label 5  --mu 1e-2 --fl_framework fedprox --bn_type bn
python proto_fssl.py --exp_name fedprox_cifar10_bn_nid --dataset cifar10 --model res9 --num_label 5  --mu 1e-2 --fl_framework fedprox --bn_type bn --non_iid
python proto_fssl.py --exp_name fedprox_svhn_bn --dataset svhn --model res9 --num_label 5  --mu 1e-4 --fl_framework fedprox --bn_type bn
python proto_fssl.py --exp_name fedprox_svhn_bn_nid --dataset svhn --model res9 --num_label 5  --mu 1e-4 --fl_framework fedprox --bn_type bn --non_iid
python proto_fssl.py --exp_name fedprox_stl10_bn --dataset stl10 --model res9 --num_label 10 --num_unlabel 980 --bn_type bn --mu 1e-2 --fl_framework fedprox

# Ablations - normalization
# FedAvg
# python proto_fssl.py --exp_name fedavg_cifar10_bn --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --bn_type bn
# python proto_fssl.py --exp_name fedavg_cifar10_bn_nid --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid --bn_type bn
# python proto_fssl.py --exp_name fedavg_cifar10_gn --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --bn_type gn
# python proto_fssl.py --exp_name fedavg_cifar10_gn_nid --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid --bn_type gn
# python proto_fssl.py --exp_name fedavg_cifar10_sbn --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --bn_type sbn
# python proto_fssl.py --exp_name fedavg_cifar10_sbn_nid --dataset cifar10 --model res9 --is_sl --num_label 54  --local_episode 2 --non_iid --bn_type sbn
# python proto_fssl.py --exp_name fedavg_cifar10_res18_sbn --dataset cifar10 --model res18 --is_sl --num_label 54  --local_episode 2 --bn_type sbn
# python proto_fssl.py --exp_name fedavg_cifar10_res18_sbn_nid --dataset cifar10 --model res18 --is_sl --num_label 54  --local_episode 2 --non_iid --bn_type sbn

# ProtoFSSL
python proto_fssl.py --exp_name cifar10_bn --dataset cifar10 --model res9 --bn_type bn
python proto_fssl.py --exp_name cifar10_bn_nid --dataset cifar10 --model res9 --non_iid --bn_type bn
python proto_fssl.py --exp_name cifar10_gn --dataset cifar10 --model res9 --bn_type gn
python proto_fssl.py --exp_name cifar10_gn_nid --dataset cifar10 --model res9 --non_iid --bn_type gn
python proto_fssl.py --exp_name cifar10_sbn --dataset cifar10 --model res9  --bn_type sbn
python proto_fssl.py --exp_name cifar10_sbn_nid --dataset cifar10 --model res9 --non_iid --bn_type sbn
python proto_fssl.py --exp_name cifar10_res18_sbn --dataset cifar10 --model res18 --bn_type sbn
python proto_fssl.py --exp_name cifar10_res18_sbn_nid --dataset cifar10 --model res18 --non_iid --bn_type sbn


# Different local episode, active clients
python proto_fssl.py --exp_name cifar10_E1 --dataset cifar10 --model res9 --local_episode 1
python proto_fssl.py --exp_name cifar10_E2 --dataset cifar10 --model res9 --local_episode 2
python proto_fssl.py --exp_name cifar10_E5 --dataset cifar10 --model res9 --local_episode 5
python proto_fssl.py --exp_name cifar10_E1_M20 --dataset cifar10 --model res9 --local_episode 1 --num_active_client 20
python proto_fssl.py --exp_name cifar10_E2_M20 --dataset cifar10 --model res9 --local_episode 1 --num_active_client 20
python proto_fssl.py --exp_name cifar10_E5_M20 --dataset cifar10 --model res9 --local_episode 5 --num_active_client 20
python proto_fssl.py --exp_name cifar10_E10_M20 --dataset cifar10 --model res9 --local_episode 10 --num_active_client 20

# Different number of unlabel samples
python proto_fssl.py --exp_name cifar10_unlabel_10 --dataset cifar10 --model res9 --q_unlabel 10
python proto_fssl.py --exp_name cifar10_unlabel_20 --dataset cifar10 --model res9 --q_unlabel 20
python proto_fssl.py --exp_name cifar10_unlabel_30 --dataset cifar10 --model res9 --q_unlabel 30
python proto_fssl.py --exp_name cifar10_unlabel_50 --dataset cifar10 --model res9 --q_unlabel 50
python proto_fssl.py --exp_name cifar10_unlabel_200 --dataset cifar10 --model res9 --q_unlabel 200
