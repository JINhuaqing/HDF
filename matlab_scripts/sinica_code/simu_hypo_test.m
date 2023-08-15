%As I have get the CV results for first two steps, I now can get the results for the third step, 
%i.e., hypothesis testing resutls


clear all; 
% always use this working directory 
%cd /data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/
cd '/Users/hujin/Library/CloudStorage/OneDrive-UCSF/Documents/ProjectCode/HDF/matlab_scripts'

addpath sinica_code/my_own/
addpath sinica_code/dantizig/
addpath sinica_code/algorithms/
addpath sinica_code/wild_bootstrap/


% parameters to use 
sn = 8;
n = 100; 
pn = 50; 
num_rep = 1000; % number of replicates
Hn = [1];

% the set of candidate parameters
save_folder = '../results/sinica_results/';
data_folder = '../mid_results/matlab_simu_data/';
data_prefix = ['PSD_sinica_d-' num2str(pn) '_n-' num2str(n) '_sn-' num2str(sn)];


% load the CV results for the first two steps
lam_results = load([save_folder data_prefix '/cv_err_eta_H1.mat']);
tau_results = load([save_folder data_prefix '/cv_err_w_H1.mat']);

% opt for lam, first step
est_diff_mat = reshape(cell2mat(lam_results.est_diffs(:))', [n, numel(lam_results.lambdas)*num_rep]);
norm_v = mean(est_diff_mat.^2);
norm_v_mat = reshape(norm_v, [num_rep, numel(lam_results.lambdas)]);
[~, ixs_opt] = min(norm_v_mat, [], 2);
opt_lams = lam_results.lambdas(ixs_opt);

% opt for tau, second step
opt_taus = tau_results.tau_v_opt;


% get the results for the third step
alpha = 0.05; % significance level
N = 10000; % number of bootstrap samples for hypothesis testing
pvals = zeros(num_rep, 1);

rep_i = 2;
%for rep_i = 1:100
    [rep_i]
    fil_name = [data_folder data_prefix '/H1_seed_' num2str(rep_i-1) '.mat'];
    cur_data = load(fil_name);
    y = cur_data.Y_centered';
    thetas = cur_data.thetas;
        
    theta1 = cell(1, pn);
    for i = 1:pn
        theta1{1, i} = squeeze(thetas(i, :, :));
    end
    
    [eta_est, f_est] = eta_est_wrapper(thetas, y, opt_lams(rep_i));
    
    M = dantizig1(Hn, theta1, opt_taus(rep_i));
    S = get_S(Hn, eta_est, M, theta1, y);
    [pval, ~, ~]= mywild(alpha, N, S);
    pvals(rep_i) = pval;
%end

save([save_folder, data_prefix '/pvals_H1.mat'], 'pvals', 'alpha', 'opt_taus', 'opt_lams')

mean(pvals(1:100)<0.05)
