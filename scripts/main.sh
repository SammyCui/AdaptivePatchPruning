#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --nodelist=node002
#SBATCH --job-name=noexadavit
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=END

module purg
module load gcc8 cuda11.2
module load openmpi/cuda/64
module load ml-pythondeps-py37-cuda11.2-gcc8/4.7.8

#RUN_NAME="evit_deit_small_369_0.5_voc_ft_lr0.00002sgd"
RUN_NAME="adavit_deit_small_369_sigma0.1_kr0.7_token_numsamples10000_voc_ft_lr0.00002adam"

source /u/erdos/cnslab/xcui32/venv/bin/activate
mkdir -p /u/erdos/cnslab/xcui32/AdaptivePatchPruning/results/$RUN_NAME
exec &> /u/erdos/cnslab/xcui32/AdaptivePatchPruning/results/$RUN_NAME/output.out


python3 /u/erdos/cnslab/xcui32/AdaptivePatchPruning/main.py \
 --root '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/' \
 --mode train --model  deit_small_patch16_adaperturbed_vit\
 --keep_rate 0.7 --prune_loc "(3,6,9)" --sigma 0.1 --decay_sigma True\
 --use_select_token True --num_samples 10000 \
 --mixup 0.8 --smoothing 0.1 --cutmix 1\
 --result_dir "/u/erdos/cnslab/xcui32/AdaptivePatchPruning/results/$RUN_NAME" \
 --write_to_collections "/u/erdos/cnslab/xcui32/AdaptivePatchPruning/results/results_vit.txt" --run_name $RUN_NAME \
 --save False --resume --pretrained True \
 --image_size 224 \
 --start_epoch 0 --max_epoch 50 --train_head_only False --per_layer_lr False\
 --lr 0.00002 --optimizer adam --lr_scheduler cosine --step_size 10 --gamma 0.2 \
 --momentum 0.9  --weight_decay 0.0005 --val_interval 1 \
 --num_workers 16 --batch_size 64 --device 'cuda:0' --download False