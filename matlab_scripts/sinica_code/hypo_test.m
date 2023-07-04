%As I have get the CV results for first two steps, I now can get the results for the third step, 
%i.e., hypothesis testing resutls


clear all; 
% always use this working directory 
cd /data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/
%cd '/Users/hujin/Library/CloudStorage/OneDrive-UCSF/Documents/ProjectCode/HDF/matlab_scripts'

addpath sinica_code/my_own/
addpath sinica_code/dantizig/
addpath sinica_code/algorithms/
addpath sinica_code/wild_bootstrap/



% the set of candidate parameters
save_folder = '../results/sinica_results/';
data_folder = '../mid_results/matlab_real_data/';
data_prefix = 'psd89_';
pn = 68;


% load the CV results for the first two steps
lam_results = load([save_folder data_prefix 'cv_err_eta.mat']);
tau_results = load([save_folder data_prefix 'cv_err_w.mat']);

n = length(lam_results.est_diffs{1}); % number of observations

% opt for lam, first step
est_diff_mat = reshape(cell2mat(lam_results.est_diffs(:))', [n, numel(lam_results.lambdas)*numel(lam_results.sns)]);
norm_v = mean(est_diff_mat.^2);
[~, ix_opt] = min(norm_v);
[opt_sn_i, opt_lam_i] = ind2sub([numel(lam_results.sns), numel(lam_results.lambdas)], ix_opt);
opt_sn = lam_results.sns(opt_sn_i);
opt_lam = lam_results.lambdas(opt_lam_i);

% opt for tau, second step
opt_tau = tau_results.tau_v_opt;


% get the data and run
fil_name = [data_folder data_prefix num2str(opt_sn) '.mat'];
cur_data = load(fil_name);
y = cur_data.Y_centered';
thetas = cur_data.thetas;
    
theta1 = cell(1, pn);
for i = 1:pn
    theta1{1, i} = squeeze(thetas(i, :, :));
end

% get the results for the third step
alpha = 0.05; % significance level
N = 10000; % number of bootstrap samples for hypothesis testing
[eta_est, f_est] = eta_est_wrapper(thetas, y, opt_lam);

pvals = zeros(pn, 1);
for fn_idx = 1:pn
    Hn = [fn_idx]; % set of hypothesis test
    Hn
    M = dantizig1(Hn, theta1, opt_tau);
    S = get_S(Hn, eta_est, M, theta1, y);
    [pval, ~, ~]= mywild(alpha, N, S);
    pvals(fn_idx) = pval;
end

save([save_folder, data_prefix 'pvals.mat'], 'pvals', 'alpha', 'N', 'S', 'eta_est', 'opt_tau', 'opt_lam', 'opt_sn')
