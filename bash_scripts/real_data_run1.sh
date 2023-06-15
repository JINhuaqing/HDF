#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=logs/HDF-%x.out
#SBATCH -J real_data89
#SBATCH --cpus-per-task=30
#SBATCH --time=48:00:00
####SBATCH --ntasks=30

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

singularity exec ~/jin/singularity_containers/hdf_latest.sif python -u /data/rajlab1/user_data/jin/MyResearch/HDF_infer/python_scripts/real_data_run1.py

