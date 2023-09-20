#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=logs/HDF-%x.out
#SBATCH -J mtest2_4
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
####SBATCH --ntasks=30

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

matlab -batch "run('/data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/sinica_code/simu_cmp2sinica_samebetaX_test2.m')"

