#!/bin/sh
#SBATCH -p gpu
#SBATCH -G v100:2
#SBATCH --mem=50G
#SBATCH -o outfile.%J
#SBATCH --mail-user=hhajj@mpiwg-berlin.mpg.de
#SBATCH -t 46:00:00
#SBATCH --nodelist=agt001
module load anaconda3
module load cuda/11.5.1
module load gcc/10.2.0
source $ANACONDA3_ROOT/etc/profile.d/conda.sh
echo $CUDA_HOME
export $CUDA_HOME=/opt/sw/rev/21.12/haswell/gcc-9.3.0/cuda-11.5.1-twasys
conda activate ml_cv
srun pip install git+https://github.com/IDEA-Research/GroundingDINO.git
srun python run_pipeline.py -i data/dataset_mod_part7 -o output/results_part7
