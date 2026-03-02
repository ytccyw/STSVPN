function [X, tnn, trank, W_new] = prox_tsvpn(Y, thresh, W, tau)
% =========================================================================
% Proximal operator for the Tensor Singular Value-Preserving Norm (TSVPN)
% This implements Theorem 1 in the manuscript.
%
% Inputs:
%   Y      - Input tensor in spatial domain
%   thresh - Threshold parameter (corresponds to 1/\mu in ADMM)
%   W      - Current weight tensor for singular values
%   tau    - Scale parameter (tau_L) for weight adjustment
%
% Outputs:
%   X      - Output tensor after weighted shrinkage
%   tnn    - Tensor nuclear norm
%   trank  - Tensor tubal rank
%   W_new  - Updated adaptive weight tensor
% =========================================================================

if (nargin < 4)
    tau = 3;   
end
 
[n1, n2, n3] = size(Y);
X = zeros(n1, n2, n3);
Y = fft(Y, [], 3);     
tnn = 0;
trank = 0;
W_new = W;        

% --- First frontal slice ---
[U, S, V] = svd(Y(:,:,1), 'econ');  
S = diag(S);       
r = length(find(S > thresh)); 

if r >= 1
    S_thresh = S(1:r) - W(1:r, 1) * thresh;    
    X(:,:,1) = U(:, 1:r) * diag(S_thresh) * V(:, 1:r)';   
    tnn = tnn + sum(S_thresh);    
    trank = max(trank, r);  
    W_new(1:r, 1) = CalcWeights_tsvpn(S_thresh, tau);
    W_new(r+1:end, 1) = 1; 
end

% --- Half of the remaining slices ---
halfn3 = round(n3 / 2);
for i = 2 : halfn3
    [U, S, V] = svd(Y(:,:,i), 'econ');
    S = diag(S);
    r = length(find(S > thresh));
    
    if r >= 1
        S_thresh = S(1:r) - W(1:r, i) * thresh;
        X(:,:,i) = U(:, 1:r) * diag(S_thresh) * V(:, 1:r)';
        tnn = tnn + sum(S_thresh) * 2; 
        trank = max(trank, r);
        W_new(1:r, i) = CalcWeights_tsvpn(S_thresh, tau);
        W_new(r+1:end, i) = 1; 
    end
    X(:,:,n3+2-i) = conj(X(:,:,i));  
end

if mod(n3, 2) == 0
    i = halfn3 + 1;
    [U, S, V] = svd(Y(:,:,i), 'econ');
    S = diag(S);
    r = length(find(S > thresh));
    
    if r >= 1
        S_thresh = S(1:r) - W(1:r, i) * thresh;
        X(:,:,i) = U(:, 1:r) * diag(S_thresh) * V(:, 1:r)';
        tnn = tnn + sum(S_thresh);
        trank = max(trank, r);
        W_new(1:r, i) = CalcWeights_tsvpn(S_thresh, tau);
        W_new(r+1:end, i) = 1;
    end
end

tnn = tnn / n3;   
X = ifft(X, [], 3); 
end