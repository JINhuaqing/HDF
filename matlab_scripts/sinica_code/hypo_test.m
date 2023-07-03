%As I have get the CV results for first two steps, I now can get the results for the third step, 
%i.e., hypothesis testing resutls


clear all; 
% always use this working directory 
cd /data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/

addpath 'sinica_code/wild bootstrap/'
addpath sinica_code/my_own/
addpath sinica_code/dantizig/



% the set of candidate parameters
sns = [6 8 10 12 14];
lambdas = [0.01, 0.03, 0.09, 0.27, 0.81, 2.73];
taus = [0.09, 0.27, 0.81, 2.73, 8.1, 24];


% load the CV results for the first two steps
save_folder = '../results/sinica_results/psd40/';
lam_results = load([save_folder 'cv_err_eta.mat']);
tau_results = load([save_folder 'cv_err_eta_w.mat']);

% opt for lam, first step
est_diff_mat = reshape(cell2mat(lam_results.est_diffs(:))', [152, numel(lam_results.lambdas)*numel(lam_results.sns)]);
norm_v = mean(est_diff_mat.^2);
[~, ix_opt] = min(norm_v);
[opt_sn_i, opt_lam_i] = ind2sub([numel(lam_results.sns), numel(lam_results.lambdas)], ix_opt);
opt_sn = lam_results.sns(opt_sn_i);
opt_lam = lam_results.lambdas(opt_lam_i);