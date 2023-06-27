clear all; 
addpath algorithms/
addpath dantizig/
addpath 'wild bootstrap'/

load ../../data/matlab_data/psd40_16.mat

pn = size(thetas, 1); % num of ROIs
sn = size(thetas, 3); % num of basis
n = size(thetas, 2); % sample size 
y = Y_centered';
lambda = 0.1; 
G=sort(repmat(1:pn,1,sn));


theta1 = cell(1, pn);
for i = 1:pn
    theta1{1, i} = squeeze(thetas(i, :, :));
end

theta2 = cell(1, pn);
for i = 1:pn
    theta2{1, i} = theta1{i}*(inv(theta1{i}'*theta1{i}))*theta1{i}';
end

theta3 = cell(1, pn);
for i = 1:pn
    theta3{1, i} = (inv(theta1{i}'*theta1{i}))*theta1{i}';
end

eta = zeros(pn*sn,1);
f = cell(1, pn);
for i = 1:pn
    f{1, i} = zeros(n, 1);
end

[eta_est, f_est] = algoscad(y, n, sn, pn, lambda, G, theta1, theta2, theta3, eta, f);

Hn = [1, 3];
lambdaseq = [0.01:0.05:0.1];
GK1=crossvalind('Kfold', n, 2);
[M, tauopt] = dantizig2(Hn, theta1, n, sn, lambdaseq, GK1);

%Hn = [1, 3];
HnC = setdiff(1:pn, Hn);

% get E cell
E = cell(1, n); 
Hn_theta1 = theta1(1, Hn);
Hn_theta1_mat = cat(2, Hn_theta1{:});
for i  = 1:n
    E{1, i} = Hn_theta1_mat(i, :)';
end

F = cell(1, n); 
HnC_theta1 = theta1(1, HnC);
HnC_theta1_mat = cat(2, HnC_theta1{:});
for i  = 1:n
    F{1, i} = HnC_theta1_mat(i, :)';
end

Lambda = cell(1, pn);
for i = 1:pn
    mat = theta1{1, i}.^2;
    Lambda{1, i} = mean(mat, 1);
end
Lambda_Hn = Lambda(1, Hn);
Lambda_Hn_vec = cell2mat(Lambda_Hn);

% select HnC from eta_est
eta_est_HnC = cell(1, size(HnC, 2));
for i = 1:length(HnC)
    eta_est_HnC{1, i} = eta_est(G==HnC(i))';
end
eta_est_HnC_vec = cell2mat(eta_est_HnC);

% Get  S
S = cell(1, n);
for i = 1:n
    right_part = Y_centered(i) - eta_est_HnC_vec * F{1, i};
    left_part = diag(1./Lambda_Hn_vec) * (M*F{1, i} - E{1, i});
    S{1, i} = left_part * right_part * n.^(-1/2);

 end
 
 % Hypo test
 [res, Tinf, CV] = wild2(0.05, n, 100, S);