#!/bin/bash
#
#SBATCH --job-name=misleader-dataset-precompute-all-embeddings
#SBATCH --mail-user=your_mail
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_model:a180"

source /miniconda3/etc/profile.d/conda.sh
conda activate lying_charts

srun python "misviz/src/model_tuning/01_precomputation/01_precompute_all_img_encodings.py" \
    --models "tinychart" \
    --datasets "misviz" \
    --datasetpaths "misviz/data/misviz/" \
    --outputpath "misviz/data/precomp/" \
    --batchsize 16