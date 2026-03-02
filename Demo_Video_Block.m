clear; clc; close all;
rng('default'); rng(2026);

current_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(current_dir, 'Functions')));
addpath(genpath(fullfile(current_dir, 'data'))); 

load('mobile_qcif_mr0p70.mat', 'Ovideo', 'Nvideo', 'Omega'); 

Ovideo = double(Ovideo);
Nvideo = double(Nvideo);
Omega = double(Omega);
[n1, n2, n3, numFrames] = size(Ovideo);

opts_tc = struct('mu', 1e-5, 'rho', 1.2, 'DEBUG', 0, 'directions', [1, 2, 4]);

tic;

disp('>>> Running STSVPN ...');
[Result_STSVPN, ~, ~, ~, ~] = tc_stsvpn(Nvideo, Omega, opts_tc);
Time_STSVPN = toc;
maxP = max(Ovideo(:));
Result_STSVPN = max(0, min(maxP, Result_STSVPN));

[psnr_noisy, ssim_noisy, fsim_noisy] = Evaluate_Video(Ovideo, Nvideo);
[psnr_sts, ssim_sts, fsim_sts] = Evaluate_Video(Ovideo, Result_STSVPN);


fprintf(' %8s    %8s    %8s    %8s\n', 'Method', 'PSNR', 'SSIM', 'FSIM');
fprintf('------------------------------------------------------\n');
fprintf(' %8s    %8.3f    %8.3f    %8.3f\n', 'Masked', psnr_noisy, ssim_noisy, fsim_noisy);
fprintf(' %8s    %8.3f    %8.3f    %8.3f\n', 'STSVPN', psnr_sts, ssim_sts, fsim_sts);
fprintf(' >>> Time: %.2f s\n\n', Time_STSVPN);


show_frame = 33;
if show_frame > numFrames
    show_frame = round(numFrames / 2);
end

frame_GT_ycbcr = Ovideo(:, :, :, show_frame) / maxP;
frame_STSVPN_ycbcr = Result_STSVPN(:, :, :, show_frame) / maxP;
current_Omega = Omega(:, :, :, show_frame);

frame_GT_RGB = ycbcr2rgb(frame_GT_ycbcr);
frame_STSVPN_RGB = ycbcr2rgb(frame_STSVPN_ycbcr);

frame_Masked_RGB = frame_GT_RGB .* current_Omega;

figure('Name', sprintf('Video Block Mask Completion (Frame %d)', show_frame), 'Position', [100, 100, 1000, 400]);
subplot(1,3,1); imshow(frame_GT_RGB); title('Original (GT)', 'FontSize', 12);
subplot(1,3,2); imshow(frame_Masked_RGB); title('Masked', 'FontSize', 12);
subplot(1,3,3); imshow(frame_STSVPN_RGB); title(sprintf('STSVPN'), 'FontSize', 12);
