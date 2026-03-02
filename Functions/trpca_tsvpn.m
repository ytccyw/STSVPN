function [L, E, obj, err, iter, W_L] = trpca_tsvpn(X, lambda, opts)

tol = 1e-6;        
max_iter = 200; 
rho = 1.1;         
mu = 1e-4;         
max_mu = 1e10;    
DEBUG = 0;
tau_L = 2;        
  
if nargin < 3, opts = []; end    
if isfield(opts, 'tol'),         tol = opts.tol;              end
if isfield(opts, 'max_iter'),    max_iter = opts.max_iter;    end
if isfield(opts, 'rho'),         rho = opts.rho;              end
if isfield(opts, 'mu'),          mu = opts.mu;                end
if isfield(opts, 'max_mu'),      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG'),       DEBUG = opts.DEBUG;          end
if isfield(opts, 'tau_L'),       tau_L = opts.tau_L;          end

dim = size(X);
L = zeros(dim);
E = zeros(dim);
Y = zeros(dim);    
d = min(dim(1), dim(2)); 
W_L = ones(d, dim(3)); 

for iter = 1 : max_iter
    Lk = L;
    Ek = E;
    
    % update L
    [L, tnnL, ~, W_L] = prox_tsvpn(X - E - Y/mu, 1/mu, W_L, tau_L);   
    
    % update E 
    E = prox_l1(X - L - Y/mu, lambda/mu);   
    
    dY = L + E - X;
    chgL = max(abs(Lk(:) - L(:)));
    chgE = max(abs(Ek(:) - E(:)));
    chg = max([chgL, chgE, max(abs(dY(:)))]); 

    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            obj = tnnL + lambda * norm(E(:), 1);
            err = norm(dY(:));
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err)]); 
        end
    end
    
    if chg < tol
        break;
    end 
    
    % update Y mu
    Y = Y + mu * dY;
    mu = min(rho * mu, max_mu);    
end
obj = tnnL + lambda * norm(E(:), 1);
err = norm(dY(:));
end