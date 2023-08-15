% Step 2
% This file is to run the CV for dantzig selector
% The default path is always matlab_scripts
% to estimate w 
% Note that we do not need to opt for sn, as the first step, we already get a opt sn.
% We just use that one

cd /home/hujin/jin/MyResearch/HDF_infer/matlab_scripts
%cd '/Users/hujin/Library/CloudStorage/OneDrive-UCSF/Documents/ProjectCode/HDF/matlab_scripts'
clear all; 
addpath sinica_code/algorithms/
addpath sinica_code/dantizig/
addpath sinica_code/wild_bootstrap/
addpath sinica_code/my_own/

% parameters to use 
sn = 8;
n = 100; 
pn = 50; 
num_rep = 1000; % number of replicates

save_folder = '../results/sinica_results/';
data_folder = '../mid_results/matlab_simu_data/';
data_prefix = ['PSD_sinica_d-' num2str(pn) '_n-' num2str(n) '_sn-' num2str(sn)];



ncv = 5; % number of folds for CV
taus= [0.09, 0.27, 0.81, 2.73, 8.1, 24]; % tau sequence, in fact it is tau in paper (3.9)
Hn = [1]; % The set of fns chosen for hypothesis testing
cv_idxs = gen_cv_idxs(n, ncv); % generate the CV index

tau_v_opt = zeros(num_rep, 1);

% parallel runing
parpool(25);

parfor rep_i = 1:num_rep
    [rep_i]
    
    fil_name = [data_folder data_prefix '/H1_seed_' num2str(rep_i-1) '.mat'];
    cur_data = load(fil_name);
    y = cur_data.Y_centered';
    thetas = cur_data.thetas;
    
    theta1 = cell(1, pn);
    for i = 1:pn
        theta1{1, i} = squeeze(thetas(i, :, :));
    end
     
    [tauopt, ~, ~] = dantizig2(Hn, theta1, taus, cv_idxs);
    tau_v_opt(rep_i) = tauopt;
      
end

saved_name = [save_folder data_prefix '/cv_err_w_H1.mat'];saved_name
save(saved_name, "taus", "tau_v_opt");
delete(gcp('nocreate'));


