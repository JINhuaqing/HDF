#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=gpu # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=scs/logs/Linear-%x-%j.out
#SBATCH -J Sn2a_Lin
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/hujin/jin/MyResearch/HDF_infer/bash_scripts/
####SBATCH --ntasks=30

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

for cv in 0.00 0.20 0.40; do
    singularity exec ~/jin/singularity_containers/hdf_ball.sif python -u ../python_scripts/simu_settingns.py --cs $cv --setting n2a
done
