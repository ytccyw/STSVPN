function w = CalcWeights_tsvpn(sigma, tau)

if (nargin < 2)
    tau = 1;
end

% Ensure singular values are strictly positive
sigma = abs(sigma);    

% Update the dynamic scale parameter (C_tau * median)
tau2 = tau * median(sigma(:)) + eps; 

% Update the concave adaptive weights
w = 1 ./ (sqrt(1 + sigma ./ tau2));   

end