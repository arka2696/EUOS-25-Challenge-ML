#!/bin/bash
#SBATCH --job-name=EuosClassicalv3
#SBATCH --partition=zen5_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

module purge
module load Mamba
source $EBROOTMAMBA/etc/profile.d/conda.sh

cd /scratch/brussel/vo/000/bvo00026/vsc11013/Projects/EUOS25

SIF=/scratch/brussel/vo/000/bvo00026/vsc11013/Projects/EUOS25/singularity/euos25_classical.sif

apptainer exec -B "$PWD":/work "$SIF" python /work/classical_runner.py
