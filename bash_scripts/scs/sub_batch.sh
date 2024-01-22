for cv in 0.1 ; do
    for setting in n1 n1a n2 n2a n3 n3a; do 
            job_name=S${setting}_c${cv}_logi
            sbatch --job-name=$job_name simu_logi_settingns_batch.sh $cv $setting
    done
done
