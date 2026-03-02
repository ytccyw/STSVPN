function [psnr_val, ssim_val, fsim_val] = Evaluate_RGB(Img1, Img2)
% =========================================================================
% Evaluate Image Quality Metrics (PSNR, SSIM, FSIM)
% =========================================================================
    maxP = max(abs(Img1(:))); 
    
    psnr_val = Calculate_PSNR(Img1, Img2, maxP);
    
    img1_gray = 255 * double(rgb2gray(Img1));
    img2_gray = 255 * double(rgb2gray(Img2));
    
    ssim_val = Calculate_SSIM(img1_gray, img2_gray);
    fsim_val = FeatureSIM(img1_gray, img2_gray);
end