% Step 1
% To estimate eta and use CV to select sn and lambda, to estimate eta
% sn = [6, 8, 10, 12, 14]
% lambdas = []

% run CV to get optimizal sn and lambda for estimate eta

clear all; 
% always use this working directory 
cd /data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/
%cd /Users/hujin/ProjectCode/HDF/matlab_scripts/
addpath sinica_code/
addpath sinica_code/algorithms/
addpath sinica_code/my_own/

% parameters to use 
sn = 8;
n = 100; 
pn = 50; 
num_rep = 1000; % number of replicates

save_folder = '../results/sinica_results/';
data_folder = '../mid_results/matlab_simu_data/';
data_prefix = ['PSD_sinica_d-' num2str(pn) '_n-' num2str(n) '_sn-' num2str(sn)];
ncv = 5; % num of CV folds

lambdas = [0.01, 0.03, 0.09, 0.27, 0.81, 2.73];

% only one loop
all_coms = num_rep * numel(lambdas);

% parallel runing
parpool(25);




est_diffs = cell(num_rep * length(lambdas), 1);
parfor ix= 1:all_coms
    [rep_i, lam_i] = ind2sub([num_rep, numel(lambdas)], ix);
    lambda = lambdas(lam_i);
    [rep_i,lambda]

    fil_name = [data_folder data_prefix '/H1_seed_' num2str(rep_i-1) '.mat'];
    cur_data = load(fil_name);
    y = cur_data.Y_centered';
    thetas = cur_data.thetas;
     
    y_est = cv_lambda_fn(thetas, y, ncv, lambda);
    
    est_diffs{ix} = y - y_est;
  
end
saved_name = [save_folder, data_prefix '/cv_err_eta_H1.mat'];saved_name
save(saved_name, "est_diffs", "num_rep", "lambdas");

% delete the pool
delete(gcp('nocreate'));



