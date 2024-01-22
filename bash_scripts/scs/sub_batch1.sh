for cv in 0.0 0.1 0.2 0.4; do
    for setting in ns1 ns1a ns1b ns1e ns2 ns2a ns2b ns2e;  do 
            job_name=S${setting}_c${cv}_linear
            sbatch --job-name=$job_name simu_settingnss_batch.sh $cv $setting
    done
done
