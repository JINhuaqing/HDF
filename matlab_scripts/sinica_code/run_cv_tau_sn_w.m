% Step 2
% This file is to run the CV for dantzig selector
% The default path is always matlab_scripts
% to estimate w 
% Note that we do not need to opt for sn, as the first step, we already get a opt sn.
% We just use that one

%cd '/Users/hujin/Library/CloudStorage/OneDrive-UCSF/Documents/ProjectCode/HDF/matlab_scripts'
cd /home/hujin/jin/MyResearch/HDF_infer/matlab_scripts
%cd matlab_scripts/
clear all; 
addpath sinica_code/algorithms/
addpath sinica_code/dantizig/
addpath sinica_code/'wild bootstrap'/

data_folder = '../data/matlab_data/';
save_folder = '../results/sinica_results/psd40/';

% get the opt sn from first step
lam_results = load([save_folder 'cv_err_eta.mat']);
% opt for lam, first step
est_diff_mat = reshape(cell2mat(lam_results.est_diffs(:))', [152, numel(lam_results.lambdas)*numel(lam_results.sns)]);
norm_v = mean(est_diff_mat.^2);
[~, ix_opt] = min(norm_v);
[opt_sn_i, opt_lam_i] = ind2sub([numel(lam_results.sns), numel(lam_results.lambdas)], ix_opt);
opt_sn = lam_results.sns(opt_sn_i);
%opt_lam = lam_results.lambdas(opt_lam_i);
opt_sn


n = 152; % number of observations
pn = 68; % number of predictors
ncv = 5; % number of folds for CV
taus= [0.09, 0.27, 0.81, 2.73, 8.1, 24]; % tau sequence, in fact it is tau in paper (3.9)
%Hn = [1]; % The set of fns chosen for hypothesis testing
cv_idxs =crossvalind('Kfold', n, ncv); % generate the CV index

tau_v_opt = zeros(pn, 1);

% parallel runing
parpool(10);

parfor cur_fn_ix = 1:pn
    Hn = [cur_fn_ix];
    [Hn]
    
    fil_name = [data_folder 'psd40_' num2str(opt_sn) '.mat'];
    cur_data = load(fil_name);
    y = cur_data.Y_centered';
    thetas = cur_data.thetas;
    
    theta1 = cell(1, pn);
    for i = 1:pn
        theta1{1, i} = squeeze(thetas(i, :, :));
    end
     
    [tauopt, ~, ~] = dantizig2(Hn, theta1, n, opt_sn, taus, cv_idxs);
    tau_v_opt(cur_fn_ix) = tauopt;
      
end

saved_name = [save_folder 'cv_err_eta_w.mat'];
save(saved_name, "taus", "tau_v_opt");
delete(gcp('nocreate'));

