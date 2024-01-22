#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=scs/logs/Logi-%x-%j.out
#SBATCH -J Sn3clogi_SIS1
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/hujin/jin/MyResearch/HDF_infer/bash_scripts/
####SBATCH --ntasks=30

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

setting=n3c
for cv in 0.00 0.20 0.40; do
    singularity exec ~/jin/singularity_containers/hdf_ball.sif python -u ../python_scripts/simu_logi_settingns_SIS1.py --cs $cv --setting $setting
done
