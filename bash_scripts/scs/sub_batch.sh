for cv in 0.0 0.1 0.2 0.4; do
    for setting in cmpn1  cmpn2 cmpn1b cmpn2b ; do 
        job_name=S${setting}_c${cv}
        sbatch --job-name=$job_name simu_settingcmpns.sh $cv $setting
    done
done
