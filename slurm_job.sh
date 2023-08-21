#!/bin/sh
#SBATCH --mem=100G
#SBATCH -p fat
#SBATCH -n 2
#SBATCH -o outfile.%J
#SBATCH --mail-user=hhajj@mpiwg-berlin.mpg.de
#SBATCH -t 24:00:00
module load anaconda3
module load cuda/11.5.1
source $ANACONDA3_ROOT/etc/profile.d/conda.sh
echo $CUDA_HOME
conda activate ml_cv
srun python run_pipeline.py -i data/dataset_mod
