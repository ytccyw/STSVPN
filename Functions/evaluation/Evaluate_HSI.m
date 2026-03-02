function [psnr_val, ssim_val, fsim_val, ergas, msam] = Evaluate_HSI(imagery1, imagery2)
% =========================================================================
% Evaluate Hyperspectral Image Quality Metrics
% =========================================================================
    [m, n, k] = size(imagery1);
    [mm, nn, kk] = size(imagery2);
    m = min(m, mm);
    n = min(n, nn);
    k = min(k, kk);
    imagery1 = imagery1(1:m, 1:n, 1:k);
    imagery2 = imagery2(1:m, 1:n, 1:k);

    psnr_val = 0;
    ssim_val = 0;
    fsim_val = 0;
    
    for i = 1:k
        img1_i = imagery1(:, :, i);
        img2_i = imagery2(:, :, i);
        
        
        mse_val = mean((img1_i(:) - img2_i(:)).^2);
        if mse_val > 0
            psnr_val = psnr_val + 10 * log10((255^2) / mse_val);
        else
            psnr_val = psnr_val + Inf; 
        end
        
     
        ssim_val = ssim_val + Calculate_SSIM(img1_i, img2_i);
        fsim_val = fsim_val + FeatureSIM(img1_i, img2_i);
    end
    
    psnr_val = psnr_val / k;
    ssim_val = ssim_val / k;
    fsim_val = fsim_val / k;
    

    ergas = ErrRelGlobAdimSyn(imagery1, imagery2);
    
    sum_sam = 0;
    for i = 1:m
        for j = 1:n
           T = imagery1(i, j, :);
           T = T(:)';
           H = imagery2(i, j, :);
           H = H(:)';
           sum_sam = sum_sam + SAM(T, H);
        end
    end
    msam = sum_sam / (m * n);
end