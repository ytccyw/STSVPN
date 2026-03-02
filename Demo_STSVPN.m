clear; clc; close all;
rng('default'); rng(2026);
current_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(current_dir, 'Functions')));
addpath(genpath(fullfile(current_dir, 'data'))); 

load('271031.mat', 'img');
X = double(img) / 255; 
[n1, n2, n3] = size(X);
Xn = zeros(size(X));
sparse_rate = 0.2;

for ii = 1:n3
    Xn(:,:,ii) = imnoise(X(:,:,ii), 'salt & pepper', sparse_rate);
end

% opts
opts = struct('mu', 1e-4, 'rho', 1.1, 'DEBUG', 0);
lambda = 1 / sqrt(max(n1, n2) * n3);

[Xhat, ~] = trpca_tsvpn(Xn, lambda, opts); 
[Xhat2, ~] = trpca_stsvpn(Xn, lambda, opts); 

[psnr_noisy, ssim_noisy, fsim_noisy] = Evaluate_RGB(X, Xn);
[psnr1, ssim1, fsim1] = Evaluate_RGB(X, Xhat);
[psnr2, ssim2, fsim2] = Evaluate_RGB(X, Xhat2);

fprintf(' %8s    %8s    %8s    %8s\n', 'Method', 'PSNR', 'SSIM', 'FSIM');
fprintf('--------------------------------------------------\n');
fprintf(' %8s    %8.3f    %8.3f    %8.3f\n', 'Noisy', psnr_noisy, ssim_noisy, fsim_noisy);
fprintf(' %8s    %8.3f    %8.3f    %8.3f\n', 'TSVPN', psnr1, ssim1, fsim1);
fprintf(' %8s    %8.3f    %8.3f    %8.3f\n', 'STSVPN', psnr2, ssim2, fsim2);

figure;
subplot(2,2,1); imshow(X / max(X(:)), []); title('Original');
subplot(2,2,2); imshow(Xn / max(Xn(:)), []); title('Noisy');
subplot(2,2,3); imshow(Xhat / max(Xhat(:)), []); title('TSVPN');
subplot(2,2,4); imshow(Xhat2 / max(Xhat2(:)), []); title('STSVPN');
