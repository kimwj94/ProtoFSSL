#!/bin/bash
#SBATCH --job-name=protofssl_stl # Submit a job named "example"
#SBATCH --nodes=1 # Using 1 node
#SBATCH --gres=gpu:1 # Using 2 GPU
#SBATCH --time=0-12:00:00 # 6 hours timelimit
#SBATCH --mem=24000MB # Using 24GB memory
#SBATCH --cpus-per-task=16 #using 16 cpus per task (srun)

source /home/${USER}/.bashrc # Initiate your shell environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate protofl # Activate your conda environment
srun python /home/keondopark/ProtoFSSL/proto_fssl.py --exp_name stl10_bn --dataset stl10 --model res9 --num_label 10 --num_unlabel 980 --bn_type bn
