% Step 1
% To estimate eta and use CV to select sn and lambda, to estimate eta
% sn = [6, 8, 10, 12, 14]
% lambdas = []

% run CV to get optimizal sn and lambda for estimate eta

clear all; 
% always use this working directory 
cd /data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/
addpath sinica_code/
addpath sinica_code/algorithms/
addpath sinica_code/my_own/

% parameters to use 
save_folder = '../results/sinica_results/';
data_folder = '../mid_results/matlab_real_data/';
data_prefix = 'psd89_';
ncv = 5; % num of CV folds

sns = [6 8 10 12 14];
lambdas = [0.01, 0.03, 0.09, 0.27, 0.81, 2.73];

% only one loop
all_coms = numel(sns) * numel(lambdas);

% parallel runing
parpool(20);




est_diffs = cell(length(sns) * length(lambdas), 1);
parfor ix= 1:all_coms
    [sn_i, lam_i] = ind2sub([numel(sns), numel(lambdas)], ix);
    sn = sns(sn_i);
    lambda = lambdas(lam_i);
    [sn,lambda]

    fil_name = [data_folder data_prefix num2str(sn) '.mat'];
    cur_data = load(fil_name);
    y = cur_data.Y_centered';
    thetas = cur_data.thetas;
     
     y_est = cv_lambda_fn(thetas, y, ncv, lambda);
    
     est_diffs{ix} = y - y_est;
  
end
saved_name = [save_folder, data_prefix 'cv_err_eta.mat'];
save(saved_name, "est_diffs", "sns", "lambdas");
   % delete the pool
delete(gcp('nocreate'));



