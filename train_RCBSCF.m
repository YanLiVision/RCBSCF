function [g_f, gf_pre] = train_RCBSCF(gf_pre,  xf, xf_pre, yf, ...
     params, use_z, frame, w, xf_r)

    for k = 1: 1
        
        % intialisation
        h_f = single(zeros(size(xf)));
        eta_f = single(zeros(size(xf)));
        h = real(ifft2(h_f));
        mu  = params.init_penalty_factor;          
        mu_scale_step = params.penalty_scale_step; 
        gamma1 = params.gamma1;
        gamma2 = params.gamma2;
        gamma3 = params.gamma3;
        
        % pre-compute the variables
        T = prod(use_z);

        Sxy =  xf.* conj(yf);
        Smx_pre = xf_pre .* conj(xf_pre);       
        Smx = xf_pre .* conj(xf);
        
        Sxx = xf .* conj(xf);
        Sxx_pre = xf_pre .* conj(xf_pre);
                                     
        Sxx_r = xf_r .* conj(xf_r);
        
        sz = size(xf);
        N = sz(1) * sz(2);
        
        % solve via ADMM algorithm
        iter = 1;
        while (iter <= params.max_iterations)

            %   solving g_f, which is equal to the filter that is to be learned         
            g_f = (Sxy + (1) * mu * h_f - (1) * eta_f + ( gamma1 * Smx_pre .* gf_pre  +  (gamma2 * Smx)  .* gf_pre )) ./ ...
                ((1 + gamma2) * Sxx + gamma1 * Sxx_pre + (1) * (mu) + gamma3 * Sxx_r);
            
            h_f = fft2(real(ifft2(mu * g_f + eta_f, 'symmetric') ./ (1/N * w.^2 + mu)));


            %   update eta
            eta_f = eta_f + (mu * (g_f - h_f));
                      
            %   update mu
            mu = min(mu_scale_step * mu, params.mu_max);
            
            iter = iter+1;
        end
        
        % save the trained filters
        gf_pre = g_f;
    end  
    
end

