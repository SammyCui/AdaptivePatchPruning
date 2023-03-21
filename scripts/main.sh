#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --nodelist=node002
#SBATCH --job-name=vit_voc20
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=END

module purg
module load gcc8 cuda11.2
module load openmpi/cuda/64
module load ml-pythondeps-py37-cuda11.2-gcc8/4.7.8

# RUN_NAME="adaperturbed_vit_369_0.7_nofreeze"
RUN_NAME="evit_deit_small_369_0.7_nofreeze_voc"

source /u/erdos/cnslab/xcui32/venv/bin/activate
mkdir -p /u/erdos/cnslab/xcui32/AdaptivePatchPruning/results/$RUN_NAME
exec >> /u/erdos/cnslab/xcui32/AdaptivePatchPruning/results/$RUN_NAME/$RUN_NAME.out


python3 /u/erdos/cnslab/xcui32/AdaptivePatchPruning/main.py \
 --root '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/' \
 --train True \
 --mode train --model deit_small_patch16_224_shrink_base \
 --keep_rate 0.7 --prune_loc "(3,6,9)" \
 --result_dir "/u/erdos/cnslab/xcui32/EfficientVideoRec/results/$RUN_NAME" \
 --write_to_collections "/u/erdos/cnslab/xcui32/EfficientVideoRec/results/results_vit.txt" --run_name $RUN_NAME \
 --save False --resume --pretrained True \
 --image_size 224 \
 --start_epoch 0 --max_epoch 50 --subset_data False \
 --lr 0.000125 --optimizer adam --lr_scheduler cosine --step_size 20 --gamma 0.2 \
 --momentum 0.9  --weight_decay 0.0005 --val_interval 1 \
 --num_workers 16 --batch_size 128 --device 'cuda:0' --download False