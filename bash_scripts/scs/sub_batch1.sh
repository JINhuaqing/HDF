for cv in 0.0 0.1 0.2 0.4; do
    for setting in cmpns1  cmpns2 cmpns1b cmpns2b ; do 
        job_name=S${setting}_c${cv}
        sbatch --job-name=$job_name simu_settingcmpnss.sh $cv $setting
    done
done
