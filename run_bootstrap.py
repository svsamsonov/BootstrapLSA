import numpy as np
import copy
from multiprocessing import Pool
import multiprocessing
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle 
import time
from tqdm import tqdm

#aux functions
def init_pi(N_a,N_s):
    """
    function to generate policy,
    inputs:
        N_a - number of actions;
        N_s - number of states;
    outputs:
        pi(a|s) - np.array of shape N_a x N_s
    """
    np.random.seed(1453)
    Pi_matr = np.random.uniform(0.0,1.0,(N_a,N_s))
    norm_coef = Pi_matr.sum(axis=0)
    Pi_matr = Pi_matr / norm_coef.reshape((1,N_s))
    #check if stochastic
    #print(Pi_matr.sum(axis=0))
    return Pi_matr


def generate_dynamics(N_a,N_s,b):
    """
    function to generate transition probabilities,
    inputs:
        N_a - number of actions;
        N_s - number of states;
        b - branching number
    outputs:
        pi(s'|s,a) - np.array of shape N_s x N_s x N_a
    """
    np.random.seed(1812)
    inds_nonzero = np.zeros((N_s,N_a,b),dtype = int)
    for i in range(N_s):
        for j in range(N_a):
            inds_nonzero[i,j] = np.random.choice(N_s, size=b, replace=False)
    Pi_matr = np.zeros((N_s,N_s,N_a),dtype=float)
    for i in range(N_s):
        for j in range(N_a):
            Pi_matr[inds_nonzero[i,j],i,j] = np.random.uniform(0.0,1.0,b)
    norm_coef = Pi_matr.sum(axis=0)
    Pi_matr = Pi_matr / norm_coef.reshape((1,N_s,N_a))
    return Pi_matr,inds_nonzero

def state_transitions(P,pi):
    """
    function to generate transition probabilities,
    inputs:
        P(s'|s,a) - np.array of shape N_s x N_s x N_a, transition probabilities;
        pi(a|s) - np.array of shape N_a x N_s, policy;
    outputs:
        p(s'|s) - transition probability matrix of shape (N_s,N_s)
    """
    np.random.seed(1812)
    P_s = np.zeros((N_s,N_s),dtype = float)
    for i in range(N_s):
        for j in range(N_s):
            P_s[i,j] = np.dot(P[i,j,:],pi[:,j])
    return P_s 

def init_rewards(N_a,N_s):
    """
    function to generate rewards,
    inputs:
        N_a - number of actions;
        N_s - number of states;
    outputs:
        R(a,s) - np.array of rewards (shape N_a x N_s)  
    """
    np.random.seed(1821)
    R = Pi_matr = np.random.uniform(0.0,1.0,(N_a,N_s))
    return R

#global constants
#number of actions
N_a = 2
#number of states
N_s = 10
#gamma = 0.99
gamma = 0.9
#branching factor (external parameter for Garnet)
branch = 3


#init policy matrix
Policy = init_pi(N_a,N_s)
#init transition matrix
P,Inds_nz = generate_dynamics(N_a,N_s,branch)
#init rewards
R = init_rewards(N_a,N_s)
#init state transition matrix
S_trans = state_transitions(P,Policy)

#system matrix
A = np.eye(N_s) - gamma*(S_trans.T)
#right hand side
b = np.sum(Policy*R,axis=0)
theta_star = np.linalg.inv(A) @ b

#note that in my notations they are usual (right) eigenvalues
eigvals, eigfuncs = np.linalg.eig(S_trans)
pi_states = -np.real(eigfuncs[:,0])
pi_states = pi_states/np.sum(pi_states)


#calculate covariance matrix Sigma_epsilon
Sigma_eps = np.zeros((N_s,N_s),dtype=float)
for s0 in range(N_s):
    for s in range(N_s):
        for a in range(N_a):
            eps = np.zeros(N_s,dtype=float)
            eps[s0] = R[a,s0] + gamma*theta_star[s]-theta_star[s0]
            eps_upd = eps.reshape((-1,1)) @ eps.reshape((1,-1))
            #averaging
            Sigma_eps += pi_states[s0]*Policy[a,s0]*P[s,s0,a]*eps_upd
print(Sigma_eps)

A_star = np.zeros((N_s,N_s),dtype=float)
for i in range(N_s):
    for j in range(N_s):
        A_star[i,j] = pi_states[i]*S_trans[j,i]
A_star = np.diag(pi_states) - gamma*A_star
b_star = b*pi_states
theta_star_new = np.linalg.inv(A_star) @ b_star

#Run TD(0) algorithm
N_iters = 4*10**6
v0 = 5*np.ones(N_s,dtype=float)
s0 = np.random.choice(N_s)
#step size
c0 = 4.0
alpha = np.zeros(N_iters,dtype=float)
for i in range(N_iters):
    alpha[i] = c0/np.sqrt(i+1)

#compute cumulative sums
Policy_cumulative = Policy.cumsum(axis=0)

Cumulative_state = np.zeros((branch,N_s,N_a))
for s in range(N_s):
    for a in range(N_a):
        cur_pol = P[Inds_nz[s,a],s,a]
        Cumulative_state[:,s,a] = cur_pol.cumsum()

def check_independent_last(seed,alpha,s_0,N_traj,n0,N_iters):
    N_max = 3200
    np.random.seed(seed)
    #number of intermediate estimates 
    num_estimates = len(N_iters)
    #total number of iterations: burn-in plus maximal number of iterations
    n_iters_max = np.max(N_iters)
    n_total = n0 + n_iters_max
    PR_V = np.zeros((num_estimates,N_traj,N_s),dtype=float)
    ###Main loop
    n_loops = N_traj // N_max
    for i in tqdm(range(n_loops)):
        V = 5*np.ones((N_max,N_s),dtype=float)
        ind_elem = 0
        V_cur = np.zeros_like(V)
        for N in range(n_total):
            #generate a set of indices
            s0 = np.random.choice(N_s,  N_max, replace=True, p = pi_states)
            #sample action
            #compute policy vectors
            Prob_vectors = Policy_cumulative[:,s0]
            a = (Prob_vectors < np.random.rand(1,  N_max)).argmin(axis=0)
            #sample next state
            #join state and action pair 
            Proba = Cumulative_state[:,s0,a]
            s_inds = (Proba < np.random.rand(1,  N_max)).argmin(axis=0)
            s = Inds_nz[s0,a,s_inds]
            #calculate J0
            eps = np.zeros(( N_max,N_s),dtype=float)
            eps[np.arange( N_max),s0] = R[a,s0] + gamma*V[np.arange(N_max),s]-V[np.arange(N_max),s0]
            #TD update
            V += alpha[N]*eps
            #update PR
            if N == n0:
                V_cur = V
            elif N > n0:
                V_cur = (V_cur*(N-n0) + V) / (N-n0+1)
            if N == n0 + N_iters[ind_elem]:
                #fill averaged estimates one by one
                PR_V[ind_elem,i*N_max:(i+1)*N_max,:] = V_cur
                ind_elem += 1
    return PR_V

#starting time
start_time = time.time()
#start loading parameters
seed = 2024
N_traj = 819200
res_last = []
res_pr = []
#number of iterations
N_b = 102400
N_iters = [200,400,800,1600,3200,6400,12800,25600,51200,102400,204800]

pr_iter = check_independent_last(seed + i,alpha,s0,N_traj,N_b,N_iters)
res_pr.append(pr_iter)

np.random.seed(2024)
#generate random direction
U = np.random.randn(N_s)
u = U/np.linalg.norm(U)
#compute the emirical CDF 
pr_samples = []

for i in range(len(N_iters)):
    norm_samples = np.sqrt(N_iters[i]//2)*A_star @ (res_pr[i] - theta_star.reshape((1,-1))).transpose()
    pr_samples.append(norm_samples)

sample = []
for i in range(len(N_iters)):
    err_proj = pr_samples[i].transpose() @ u
    sample.append(err_proj)
sample = np.asarray(sample)

sample_pickl = {'description':"Experiments 18.05.2024, step size k^{-1/2}, 819200 trajectories",'samle size':N_iters,'sample':sample}
with open('res_18_05.pickle', 'wb') as handle:
    pickle.dump(sample_pickl, handle)

all_ks = []
loc = 0.0
scale = (Sigma_eps @ u).dot(u)
for i in range(len(N_iters)):
    sample_sort = np.sort(sample[i])
    ecdf = np.linspace(0,1,N_traj+1)
    cdfs = norm.cdf(sample_sort, loc = 0, scale = np.sqrt(scale))
    ks_dist = max([(ecdf[1:] - cdfs).max(), (cdfs - ecdf[:-1]).max()])
    all_ks.append(ks_dist)

print("total time elapsed: ",time.time()-start_time)
