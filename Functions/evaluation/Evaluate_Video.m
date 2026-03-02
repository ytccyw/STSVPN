function [avg_psnr, avg_ssim, avg_fsim] = Evaluate_Video(Video1, Video2)
% =========================================================================
% Evaluate Video Sequence Quality Metrics
% =========================================================================
    [height, width, channels, numFrames] = size(Video1);
    [height2, width2, channels2, numFrames2] = size(Video2);
    height = min(height, height2);
    width = min(width, width2);
    channels = min(channels, channels2);
    numFrames = min(numFrames, numFrames2);
    
    Video1 = Video1(1:height, 1:width, 1:channels, 1:numFrames);
    Video2 = Video2(1:height, 1:width, 1:channels, 1:numFrames);
    
    psnr_total = 0;
    ssim_total = 0;
    fsim_total = 0;
    maxP = max(abs(Video1(:)));  
    
    for t = 1:numFrames
        frame1 = Video1(:, :, :, t);
        frame2 = Video2(:, :, :, t);
        
        
        psnr_total = psnr_total + Calculate_PSNR(frame1, frame2, maxP);
        
        frame1_gray = 255 * double(rgb2gray(frame1));
        frame2_gray = 255 * double(rgb2gray(frame2));
     
        ssim_total = ssim_total + Calculate_SSIM(frame1_gray, frame2_gray);
        fsim_total = fsim_total + FeatureSIM(frame1_gray, frame2_gray);
    end
    
    avg_psnr = psnr_total / numFrames;
    avg_ssim = ssim_total / numFrames;
    avg_fsim = fsim_total / numFrames;
end
