for cv in 0.0 0.1 0.2 0.4; do
    for setting in nm1 nm1a nm1b nm1e nm2 nm2a nm2b nm2e;  do 
            job_name=S${setting}_c${cv}_linear
            sbatch --job-name=$job_name simu_settingnms_batch.sh $cv $setting
    done
done
