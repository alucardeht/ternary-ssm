import numpy as np

def compute_chebyshev_coefficients(epsilon_0=1e-4, degree=8):
    N = 1000
    k = np.arange(1, N + 1)
    y_nodes = np.cos((2 * k - 1) * np.pi / (2 * N))
    x_nodes = (y_nodes * (1 - epsilon_0) + (1 + epsilon_0)) / 2.0
    f_val = x_nodes * np.log(x_nodes)
    
    c = []
    for j in range(degree + 1):
        T_j = np.cos(j * np.arccos(y_nodes))
        factor = 1.0 / N if j == 0 else 2.0 / N
        c_j = factor * np.sum(f_val * T_j)
        c.append(c_j)
    return c

def exact_sre(K):
    # Compute exact eigenvalues
    eigenvals = np.linalg.eigvalsh(K)
    # Filter very small/negative eigenvalues to prevent log errors
    eigenvals = eigenvals[eigenvals > 0]
    tr_K = np.sum(eigenvals)
    p = eigenvals / tr_K
    sre = -np.sum(p * np.log(p))
    return sre, tr_K

def estimate_sre_cheb_hutch(K, c, epsilon_0=1e-4, S=4):
    d = K.shape[0]
    # 1. Compute lambda_max using infinity norm as bound
    # (or max row sum)
    lambda_max = np.max(np.sum(np.abs(K), axis=1))
    if lambda_max == 0:
        return 0.0
        
    K_tilde = K / lambda_max
    tr_K_tilde = np.trace(K_tilde)
    
    # 2. Stochastic estimation of Tr(f(K_tilde))
    tr_f_K_tilde_est = 0.0
    
    # Rademacher vectors
    for _ in range(S):
        v = np.random.choice([-1.0, 1.0], size=d)
        
        # Chebyshev recurrence on Y = (2 * K_tilde - (1 + eps) * I) / (1 - eps)
        # w_0 = v
        # w_1 = Y v = (2 * K_tilde v - (1 + eps) * v) / (1 - eps)
        # w_j = 2 Y w_{j-1} - w_{j-2} = (4 * K_tilde w_{j-1} - 2*(1+eps)*w_{j-1}) / (1 - eps) - w_{j-2}
        
        w_prev2 = v.copy()
        
        # w_1 calculation
        K_v = K_tilde @ v
        w_prev1 = (2.0 * K_v - (1.0 + epsilon_0) * v) / (1.0 - epsilon_0)
        
        # accumulate term for j=0 and j=1
        est_v = c[0] * w_prev2 + c[1] * w_prev1
        
        # j = 2 to 8
        for j in range(2, 9):
            K_w = K_tilde @ w_prev1
            w_j = (4.0 * K_w - 2.0 * (1.0 + epsilon_0) * w_prev1) / (1.0 - epsilon_0) - w_prev2
            est_v += c[j] * w_j
            
            w_prev2 = w_prev1.copy()
            w_prev1 = w_j.copy()
            
        tr_f_K_tilde_est += np.dot(v, est_v)
        
    tr_f_K_tilde_est /= S
    
    # Reconstruct estimated SRE
    # SRE = ln(Tr(K)) - Tr(K ln K) / Tr(K)
    # Tr(K) = lambda_max * Tr(K_tilde)
    # Tr(K ln K) = lambda_max * Tr(K_tilde * (ln K_tilde + ln(lambda_max) * I))
    #            = lambda_max * Tr(f(K_tilde)) + lambda_max * ln(lambda_max) * Tr(K_tilde)
    # Thus,
    # SRE_est = ln(lambda_max * tr_K_tilde) - (lambda_max * tr_f_K_tilde_est + lambda_max * ln(lambda_max) * tr_K_tilde) / (lambda_max * tr_K_tilde)
    #         = ln(lambda_max) + ln(tr_K_tilde) - tr_f_K_tilde_est / tr_K_tilde - ln(lambda_max)
    #         = ln(tr_K_tilde) - tr_f_K_tilde_est / tr_K_tilde
    # Look at how the lambda_max beautifully cancels out in the SRE computation!
    
    sre_est = np.log(tr_K_tilde) - tr_f_K_tilde_est / tr_K_tilde
    return sre_est

if __name__ == "__main__":
    np.random.seed(42)
    d = 128
    # Create a random positive semi-definite matrix to simulate K
    W = np.random.choice([-1.0, 0.0, 1.0], size=(512, d))
    K = W.T @ W
    
    c = compute_chebyshev_coefficients(1e-4, 8)
    
    exact_val, tr_val = exact_sre(K)
    est_val = estimate_sre_cheb_hutch(K, c, 1e-4, S=8)
    
    print(f"Exact SRE: {exact_val:.6f} (Theoretical Max: {np.log(d):.6f})")
    print(f"Estimated SRE: {est_val:.6f}")
    print(f"Absolute SRE Error: {abs(exact_val - est_val):.6f}")
