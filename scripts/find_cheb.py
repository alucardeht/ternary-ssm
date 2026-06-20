import numpy as np

def compute_chebyshev_coefficients(epsilon_0=1e-4, degree=8):
    # Map y in [-1, 1] to x in [epsilon_0, 1]
    # x = (y * (1 - epsilon_0) + (1 + epsilon_0)) / 2
    
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

def evaluate_chebyshev(x, c, epsilon_0=1e-4):
    # Map x in [epsilon_0, 1] to y in [-1, 1]
    y = (2.0 * x - (1.0 + epsilon_0)) / (1.0 - epsilon_0)
    
    # Evaluate Chebyshev polynomial
    degree = len(c) - 1
    T = [np.ones_like(y), y]
    for j in range(2, degree + 1):
        T.append(2.0 * y * T[-1] - T[-2])
        
    approx = np.zeros_like(y)
    for j, c_j in enumerate(c):
        approx += c_j * T[j]
    return approx

if __name__ == "__main__":
    epsilon_0 = 1e-4
    c = compute_chebyshev_coefficients(epsilon_0, 8)
    print(f"Chebyshev coefficients for epsilon_0 = {epsilon_0}:")
    for j, val in enumerate(c):
        print(f"c[{j}] = {val:.10f}f;")
        
    # Evaluate error
    test_x = np.linspace(epsilon_0, 1.0, 10000)
    exact = test_x * np.log(test_x)
    approx = evaluate_chebyshev(test_x, c, epsilon_0)
    
    max_error = np.max(np.abs(exact - approx))
    print(f"Max absolute error on [{epsilon_0}, 1]: {max_error:.10f}")
