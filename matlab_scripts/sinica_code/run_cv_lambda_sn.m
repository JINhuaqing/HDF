% To estimate eta and use CV to select sn and lambda
% sn = [6, 8, 10, 12, 14]
% lambdas = []

% run CV to get optimizal sn and lambda for estimate eta

clear all; 
% always use this working directory 
cd /data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/
addpath sinica_code/
addpath sinica_code/algorithms/

% parameters to use 
ncv = 5; % num of CV folds
n = 152; % num of subjects
pn = 68; % num of ROIs, fns. 

sns = [6 8 10 12 14];
lambdas = [0.01, 0.03, 0.09, 0.27, 0.81, 2.73];

% only one loop
all_coms = numel(sns) * numel(lambdas);

% parallel runing
parpool(10);




est_diffs = cell(length(sns) * length(lambdas), 1);
parfor ix= 1:all_coms
    [sn_i, lam_i] = ind2sub([numel(sns), numel(lambdas)], ix);
    sn = sns(sn_i);
    lambda = lambdas(lam_i);
    [sn,lambda]

    fil_name = ['../data/matlab_data/psd40_' num2str(sn) '.mat'];
    cur_data = load(fil_name);
    y = cur_data.Y_centered';
    thetas = cur_data.thetas;
     
     y_est = cv_lambda_fn(thetas, y, ncv, lambda);
    
     est_diffs{ix} = y - y_est;
  
end
 saved_name = ['../results/sinica_results/psd40/cv_err_eta.mat' ];
 save(saved_name, "est_diffs", "sns", "lambdas");
   % delete the pool
delete(gcp('nocreate'));



