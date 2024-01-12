#!/bin/bash
#### The job script, run it as qsub xxx.sh 

#### the shell language when run via the job scheduler [IMPORTANT]
#$ -S /bin/bash
#### job should run in the current working directory
###$ -cwd
##### set job working directory
#$ -wd  /wynton/home/rajlab/hjin/MyResearch/HDF/bash_scripts/
#### Specify job name
#$ -N Sn3
#### Output file
#$ -o wynton/logs/Logi-$JOB_NAME_$JOB_ID.out
#### Error file
#$ -e wynton/logs/Logi-$JOB_NAME_$JOB_ID.err
#### memory per core
#$ -l mem_free=2G
#### number of cores 
#$ -pe smp 40
#### Maximum run time 
#$ -l h_rt=48:00:00
#### job requires up to 2 GB local space
#$ -l scratch=2G
#### Specify queue
###  gpu.q for using gpu
###  if not gpu.q, do not need to specify it
###$ -q gpu.q 
#### The GPU memory required, in MiB
### #$ -l gpu_mem=12000M

echo "Starting running"
setting=n3

#singularity exec ~/MyResearch/hdf_orthbasis.sif python -u ../python_scripts/simu_logi_settingns.py --cs 0.00 --setting $setting
#singularity exec ~/MyResearch/hdf_orthbasis.sif python -u ../python_scripts/simu_logi_settingns.py --cs 0.20 --setting $setting
singularity exec ~/MyResearch/hdf_orthbasis.sif python -u ../python_scripts/simu_logi_settingns.py --cs 0.40 --setting $setting

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
