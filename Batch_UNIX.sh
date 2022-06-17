#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=1-00:00:00
#SBATCH --array=1
#SBATCH --output=MRF-%j.out
#SBATCH --error=MRF_error-%j.err
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user={INSERT EMAIL}

cd ~/staff-bulk/MRF_optimisation
. ~/.bashrc
conda activate environment
export OPENBLAS_NUM_THREADS=1
python main_undersampling.py
exit 0
