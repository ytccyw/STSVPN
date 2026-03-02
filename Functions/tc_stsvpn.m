function [L, obj, err, iter, W_k] = tc_stsvpn(X, Omega, opts)
% =========================================================================
% Extends STSVPN for structured missing data (e.g., Block Mask Video Completion).
% =========================================================================
    tol = 1e-6;        
    max_iter = 200;    
    rho = 1.2;     
    mu = 1e-5;         
    max_mu = 1e10;     
    DEBUG = 0;
    tau_L = 12;         
    directions = 1:2;  
      
    if nargin < 3, opts = []; end    
    if isfield(opts, 'tol'),         tol = opts.tol;              end
    if isfield(opts, 'max_iter'),    max_iter = opts.max_iter;    end
    if isfield(opts, 'rho'),         rho = opts.rho;              end
    if isfield(opts, 'mu'),          mu = opts.mu;                end
    if isfield(opts, 'max_mu'),      max_mu = opts.max_mu;        end
    if isfield(opts, 'DEBUG'),       DEBUG = opts.DEBUG;          end
    if isfield(opts, 'tau_L'),       tau_L = opts.tau_L;          end
    if isfield(opts, 'directions'),  directions = opts.directions;end
    
    if ndims(X) == 4 && isequal(directions, [1, 2, 4])
        X_perm = permute(X, [1, 2, 4, 3]);
        Omega_perm = permute(Omega, [1, 2, 4, 3]);
        L_perm = zeros(size(X_perm));
        
        opts_3D = opts;
        opts_3D.directions = [1, 2, 3]; 
        
        num_channels = size(X, 3);
        for c = 1:num_channels
            [L_3D, ~, ~, ~, ~] = tc_stsvpn(X_perm(:,:,:,c), Omega_perm(:,:,:,c), opts_3D);
            L_perm(:,:,:,c) = L_3D;
        end
        
        L = permute(L_perm, [1, 2, 4, 3]);
        obj = 0; err = 0; iter = 0; W_k = cell(0); 
        return; 
    end
    
    dim = size(X);
    L = zeros(dim);
    Z = X;             
    Y = zeros(dim);    
    d = min(dim(1), dim(2)); 
  
    num_dirs = length(directions);
    
    G = cell(1, max(directions));
    Lambda_k = cell(1, max(directions)); 
    W_k = cell(1, max(directions));     
    
    Denom_FFT_base = zeros(dim);             
    for i = 1:num_dirs
        idx = directions(i);
        G{idx} = porder_diff(L, idx); 
        Lambda_k{idx} = zeros(dim); 
        W_k{idx} = ones(d, dim(3));
        
        Denom_FFT_base = Denom_FFT_base + diff_element(dim, idx); 
    end
    Denom_FFT_base = 1 + Denom_FFT_base; 
    
    Omega_idx = (Omega == 1);
    
    % Main loop
    for iter = 1 : max_iter
        Lk = L;
        Zk = Z;
        
        % Update L 
        sum_diff_T = zeros(dim);
        for i = 1:num_dirs
           idx = directions(i);
           sum_diff_T = sum_diff_T + porder_diff_T(mu * G{idx} - Lambda_k{idx}, idx); 
        end
        
         % Update L
        L = real(ifftn(fftn(mu * Z + Y + sum_diff_T) ./ (mu * Denom_FFT_base)));
   
        % Update G_k
        tnn_sum = 0;
        for i = 1:num_dirs
            idx = directions(i);
            tensor_in = porder_diff(L, idx) + Lambda_k{idx} / mu;
            [G{idx}, tnn_val, ~, W_k{idx}] = prox_tsvpn(tensor_in, 1/(mu * num_dirs), W_k{idx}, tau_L);
            tnn_sum = tnn_sum + tnn_val;
        end
    
        % Update Z
        Z = L - Y / mu;
        Z(Omega_idx) = X(Omega_idx); 
        
        dY = Z - L;
      
        Y = Y + mu * dY;
        mu = min(rho * mu, max_mu);  
        for i = 1:num_dirs
            idx = directions(i); 
            Lambda_k{idx} = Lambda_k{idx} + mu * (porder_diff(L, idx) - G{idx});
        end
        
        chgL = max(abs(Lk(:) - L(:)));
        chgZ = max(abs(Zk(:) - Z(:)));
        chg = max([chgL, chgZ, max(abs(dY(:)))]); 
        
        if chg < tol
            break;
        end
        
        if DEBUG
            if iter == 1 || mod(iter, 10) == 0
                err = norm(dY(:));
                fprintf('Iter %d, mu = %f, err = %f\n', iter, mu, err);
            end
        end
    end
    
    obj = tnn_sum / num_dirs;
    err = norm(dY(:));
end