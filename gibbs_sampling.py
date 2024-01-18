import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad, trapz # Integration and double integration
from scipy.special import betainc, betaincinv # Useful for the normalisation coefficient in question 1
from scipy.stats import invgamma
from scipy.linalg import inv
from scipy.linalg import toeplitz
from numba import jit

l = 0
k = 100
T = 200
rho = 0.75
a, b, A, B = 1, 1, 1, 1

@jit
def density_R2(r, z):
    s = np.sum(z)
    return ((1-r)/r)**s/2

def posterior_R2(r, z):
    return quad(lambda x: density_R2(x, z), 10**(-3), r)[0]/quad(lambda x: density_R2(x, z), 10**(-3), 1 - 10**(-3))[0]

def find_nearest_idx(value, array):
    idx = np.abs(array - value).argmin()
    return idx

def draw_R2_q(z, a, b, A, B, support):
    k = len(z)
    s = np.sum(z)
    a_1 = s + s/2 + a 
    b_1 = k - s + b 
    point = np.random.uniform(0, 1)
    CDF = np.array([posterior_R2(support[i], z) for i in range(len(support))])
    q_draw = betaincinv(a_1, b_1, point)
    nearest_index_point = find_nearest_idx(point, CDF)
    R2_draw = support[nearest_index_point]
    
    return (R2_draw, q_draw)

def sample_z(Y, X, q_start, R_2_start, num_iterations, z, support):
    k = len(z)
    zaza = np.copy(z)
    R_2, q = R_2_start, q_start
    try :
        # For each iteration of the Gibbs
        for t in range(num_iterations):
            # We update our z
            for i in range(k):
                R_2, q = draw_R2_q(zaza, 1, 1, 1, 1, support)
                v = np.mean(np.var(X, axis = 0))# Mean of the variances of each predictor (axis = 0 because a row of x is a regressor)
                gamma_2 = R_2/(q*k*v*(1-R_2))

                # In order to operate P(zi = 1 | Y,X,q, R_2, z_{-i}), we prepare two versions of every vectors, 
                #for each case where z_i = 1 and z_i = 0
                zeta, zeta_0 = np.copy(zaza), np.copy(zaza)
                if zeta[i] == 1:
                    zeta_0[i] = 0
                else:
                    zeta[i] = 1

                non_zeros_index = np.nonzero(zeta)[0]
                non_zeros_index_0 = np.nonzero(zeta_0)[0]

                # Here, for exemple, we compute X_tilde for the case where the i-th cooridinate of X is null or not
                X_tilde = X[ : , non_zeros_index]
                X_tilde_0 = X[ : , non_zeros_index_0]

                # Same for W_tilde
                W_tilde = (np.dot(X_tilde.T, X_tilde) + np.eye(len(non_zeros_index))/gamma_2)
                W_tilde_0 = (np.dot(X_tilde_0.T, X_tilde_0) + np.eye(len(non_zeros_index_0))/gamma_2)

                W_tilde_inv = inv(W_tilde)
                W_tilde_inv_0 = inv(W_tilde_0)

                beta_tilde_hat = np.dot(W_tilde_inv, np.dot(X_tilde.T, Y))
                beta_tilde_hat_0 = np.dot(W_tilde_inv_0, np.dot(X_tilde_0.T, Y))

                p = 1 / (1 + ( ((gamma_2**0.5)*(1-q)) / q ) * (np.linalg.det(W_tilde_inv_0)/np.linalg.det(W_tilde_inv))**(0.5) 
                            * ( (np.dot(Y.T,Y) - np.dot(beta_tilde_hat_0.T , np.dot(W_tilde_0, beta_tilde_hat_0))) 
                                / (np.dot(Y.T,Y) - np.dot(beta_tilde_hat.T , np.dot(W_tilde, beta_tilde_hat))))**(-T//2))

                u = np.random.uniform(0, 1)
                if u <= p:
                    zaza[i] = 1
                else:
                    zaza[i] = 0
    except ValueError :
        print(z)
        return np.copy(z)
    
    return zaza