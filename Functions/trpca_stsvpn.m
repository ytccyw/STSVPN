function [L, E, obj, err, iter, W_k] = trpca_stsvpn(X, lambda, opts)
    
    tol = 1e-6;        
    max_iter = 200; 
    rho = 1.1;        
    mu = 1e-4;         
    max_mu = 1e10;    
    DEBUG = 0;
    tau_L = 2;     
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
    
    dim = size(X);
    L = zeros(dim);
    E = zeros(dim); 
    Y = zeros(dim); 
    d = min(dim(1), dim(2)); 
  
    num_dirs = length(directions);
    

    G = cell(1, max(directions));
    Lambda_k = cell(1, max(directions)); 
    W_k = cell(1, max(directions));     
    Denom_FFT = zeros(dim);             

    for i = 1:num_dirs
        idx = directions(i);
        G{idx} = porder_diff(L, idx); 
        Lambda_k{idx} = zeros(dim); 
        W_k{idx} = ones(d, dim(3));
        
        Denom_FFT = Denom_FFT + diff_element(dim, idx); 
    end
    
    % Main loop
    for iter = 1 : max_iter
        Lk = L;
        Ek = E;

        % 1. Update L 
        sum_diff_T = zeros(dim);
        for i = 1:num_dirs
           idx = directions(i);
           sum_diff_T = sum_diff_T + porder_diff_T(mu * G{idx} - Lambda_k{idx}, idx); 
        end
    
        L = real(ifftn(fftn(mu * (X - E) + Y + sum_diff_T) ./ (mu * (1 + Denom_FFT))));
   
        % 2. Update gradient tensors G_k
        for i = 1:num_dirs
            idx = directions(i);
         
            tensor_in = porder_diff(L, idx) + Lambda_k{idx} / mu;
            [G{idx}, tnnL, ~, W_k{idx}] = prox_tsvpn(tensor_in, 1/(mu * num_dirs), W_k{idx}, tau_L);
        end
    
        % 3. Update E 
        E = prox_l1(X - L + Y/mu, lambda/mu);
        
        % 4. Residual and Multipliers
        dY = X - L - E;
        
        Y = Y + mu * dY;
        mu = min(rho * mu, max_mu);  

        for i = 1:num_dirs
            idx = directions(i); 
            Lambda_k{idx} = Lambda_k{idx} + mu * (porder_diff(L, idx) - G{idx});
        end

        chgL = max(abs(Lk(:) - L(:)));
        chgE = max(abs(Ek(:) - E(:)));
        chg = max([chgL, chgE, max(abs(dY(:)))]); 
        
        if chg < tol
            break;
        end

        if DEBUG
            if iter == 1 || mod(iter, 10) == 0
                obj = tnnL + lambda * norm(E(:), 1);
                err = norm(dY(:));
                disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err)]);
            end
        end
    end

    obj = tnnL + lambda * norm(E(:), 1);
    err = norm(dY(:));
end