#!/bin/bash

#SBATCH --job-name=finetune-axisdeplot-accelerate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=your_mail
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2

######################
### Set enviroment ###
######################
source /miniconda3/etc/profile.d/conda.sh
conda activate lying_charts
export GPUS_PER_NODE=2
######################

export SCRIPT="misviz/src/model_tuning/02_deplot_finetune/deplot_finetune_accelerate_enabled.py"
export SCRIPT_ARGS=" \
    --experiment_name deplot_finetune_accelerate \
    --outputpath misviz/data/deplot_finetune/output/ \
    --datasetpath misviz/data/misviz_synth/ \
    --axis_data_path misviz/data/misviz_synth/axis_variation/ \
    --epochs 4 \
    --batch_size 4 \
    --seq_length 1024 \
    --mixed_precision no \
    --lora_on
    "

accelerate launch --num_processes $GPUS_PER_NODE $SCRIPT $SCRIPT_ARGS