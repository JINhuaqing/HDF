for cv in 0.0 0.1 0.2 0.4; do
    for setting in ns1 ns1a ns2 ns2a ; do 
        for start in 0 50 100 150; do
            job_name=S${setting}_c${cv}_${start}_logi
            sbatch --job-name=$job_name simu_logi_settingnss_batch.sh $cv $setting $start
        done
    done
done
