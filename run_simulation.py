import math
import os
import pickle
import argparse
import json

import numpy as np
import scipy.stats as stats

from utils import dict_sum

parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, default='TS', \
        choices=['TS', 'epsilon', 'independent'])
parser.add_argument('--N', type=int, default=100000, 
        help='Number of monte carlo simulations')
parser.add_argument('--n', type=int, default=100,
        help='Batch size')
parser.add_argument('--T', type=int, default=5,
        help='Number of batches')
parser.add_argument('--K', type=int, default=2, 
        help='Number of arms (code only written for K=2!)')
parser.add_argument('--means', type=str, default='0,0',
        help='Expected rewards for each arm (file name in non-stationary case)')
parser.add_argument('--var', type=float, default=1,
        help='Reward variance (sigma^2)')
parser.add_argument('--clipping', type=float, default=0.0,
        help='Clipping value in [0, 1)')
parser.add_argument('--no_zeros', type=int, default=1,
        help='Ensure that each batch has at least one sample per arm')
parser.add_argument('--reward', type=str, default='normal', \
        choices=['bernoulli', 'normal', 'uniform'])
parser.add_argument('--pi1', type=float, default=0.5,
        help='Sampling probability for first batch') 
parser.add_argument('--prior_means', type=str, default='0,0',
        help='Prior means on rewards for each arm used for Thompson Sampling')
parser.add_argument('--prior_vars', type=str, default='1,1',
        help='Prior variances on rewards for each arm used for Thompson Sampling')
parser.add_argument('--alg_var', type=float, default=1,
        help='Variance of rewards assumed by Thompson Sampling')
parser.add_argument('--save_rewards', type=int, default=1,
        help='To save time can set to 0 to not save rewards (need to estimate variance)')
parser.add_argument('--nonstationary', type=int, default=0,
        help='Whether rewards are nonstationary over batches (reads rewards from file)')

parser.add_argument('--path', type=str, default='./simulations')
parser.add_argument('--nonstationary_path', type=str, default='./nonstationary_means',
        help='Path to folder with nonstationary mean files')

args = parser.parse_args()
print(vars(args))
assert args.K == 2          # Code only written for two arms
assert args.clipping < 1
if args.strategy == 'epsilon':
    assert args.clipping > 0

nonsave_args = ['path', 'nonstationary_path']
save_args = ['{}={}'.format(key, val) for key, val in vars(args).items() if key not in nonsave_args]
save_str = '_'.join(save_args)

path = args.path
if not os.path.isdir(path):
    os.mkdir(path)
save_f = os.path.join(path, save_str)
if not os.path.isdir(save_f):
    os.mkdir(save_f)

# Save arguments
with open(save_f+'/args.json', 'w') as f:
    json.dump(vars(args), f, indent=4)

# Process expected rewards for each arm in stationary and non-stationary cases
if not args.nonstationary:
    true_means = [float(x) for x in args.means.split(',')]
    assert len(true_means) == args.K
else:
    with open( os.path.join( args.nonstationary_path, args.means), 'r') as f:
        lines = f.readlines()
    means0, means1 = lines[0], lines[1]
    true_means = [[float(x) for x in means0.split(',')], \
            [float(x) for x in means1.split(',')]]
    assert len(true_means[0]) == len(true_means[1])
    assert len(true_means[0]) == args.T

print('Arm Means:', true_means)

prior_means = [float(x) for x in args.prior_means.split(',')]
prior_vars = [float(x) for x in args.prior_vars.split(',')]
assert len(prior_means) == args.K
assert len(prior_vars) == args.K


all_sums = {i:{} for i in range(args.K)}
all_counts = {i:{} for i in range(args.K)}
all_rewards = {i:{} for i in range(args.K)}
all_residuals = {i:{} for i in range(args.K)}
all_pis = {}


def get_pis_TS(all_counts, all_sums, var, clipping=0):
    # Calculate posterior to get probability of sampling arm
    all_means = []
    summed_counts = []
    pm = []
    pv = []
    for k in range(args.K):
        counts_k = dict_sum(all_counts[k])
        summed_counts.append(counts_k)
        sums_k = dict_sum(all_sums[k])
        mean_k = np.divide(sums_k, counts_k, out=np.zeros_like(sums_k), where=counts_k!=0) 
        all_means.append(mean_k) 
       
        # Posterior mean
        pm_temp = np.divide(prior_means[k]*var + prior_vars[k]*mean_k*counts_k, var + prior_vars[k]*counts_k)
        # Posterior variance
        pv_temp = np.divide(prior_vars[k]*var, var + prior_vars[k]*counts_k)
        pm.append(pm_temp)
        pv.append(pv_temp)

    pv = np.array(pv)           # Posterior variance
    ps = np.sqrt(pv)            # Posterior std
    pm = np.array(pm)           # Posterior mean
    all_means = np.array(all_means)

    pis = []
    post_mean = pm[1] - pm[0]
    post_var = pv[1] + pv[0]
    # Calculate sampling probability
    ratio = np.divide(post_mean, np.sqrt(post_var))
    pis = stats.norm.cdf(ratio)
    
    if clipping > 0:
        pis = np.minimum(np.maximum(pis, clipping), 1-clipping)

    return pis

def get_pis_epsilon(all_counts, all_sums, clipping=0):
    all_means = []
    for k in range(args.K):
        all_means.append(np.divide(dict_sum(all_sums[k]), dict_sum(all_counts[k])))
    all_means = np.transpose(np.nan_to_num(all_means))

    pis = np.array([[clipping for k in range(args.K)] for k in range(args.N)])
    max_vals = np.broadcast_to(np.expand_dims(np.max(all_means, axis=1), 1), (args.N, args.K))
    pis += (1-2*clipping)*(np.equal(max_vals, all_means))
    return pis[:, 1]


# pis ( N x K )
pis = np.array([args.pi1 for i in range(args.N)])
all_pis = []
all_pis.append(pis)


for t in range(1, args.T+1):
    print('T={}'.format(t))

    # Sample arms
    counts_sample1 = np.random.binomial([args.n for i in range(args.N)], pis, args.N)
    if args.no_zeros:
        # Ensures that in each batch, each arm is sampled at least once
        zeros = 1*np.equal(counts_sample1, 0)
        all_n = 1*np.equal(counts_sample1, args.n)
        counts_sample1 = counts_sample1 + zeros - all_n
    counts_sample0 = args.n-counts_sample1
    # counts_sample ( N x k )
    counts_sample = np.transpose(np.array([counts_sample0, counts_sample1]))

    # Sample reward noise
    if args.reward == 'normal':
        noise = np.random.normal(0, math.sqrt(args.var), (args.N, args.n))
    elif args.reward == 'bernoulli':
        noise = math.sqrt(args.var)*2*(np.random.binomial(1, 0.5, size=(args.N, args.n)) - 0.5) 
    elif args.reward == 'uniform':
        noise = math.sqrt(args.var*12)*(np.random.uniform(0, 1, size=(args.N, args.n)) - 0.5) 

    seen_counts = np.zeros(args.N, dtype='int')
    for k in range(args.K):
        counts_k = counts_sample[:, k]
        all_counts[k][t] = counts_k
        if not args.nonstationary:
            rewards = [noise[ i,seen_counts[i] : seen_counts[i]+counts_k[i] ] + true_means[k] for i in range(args.N)]
        else:
            rewards = [noise[ i,seen_counts[i] : seen_counts[i]+counts_k[i] ] + true_means[k][t-1] for i in range(args.N)]
        if args.save_rewards:
            all_rewards[k][t] = rewards
        all_sums[k][t] = np.array([np.sum(rewards[i]) for i in range(args.N)])
        seen_counts += counts_k
    
    if t != args.T:
        if args.strategy == 'TS':
            pis = get_pis_TS(all_counts, all_sums, args.alg_var, clipping=args.clipping)
        elif args.strategy == 'epsilon':
            pis = get_pis_epsilon(all_counts, all_sums, clipping=args.clipping)
        elif args.strategy == 'no_rl':
            pis = 0.5*np.ones(args.N)
        else:
            raise ValueError('Invalid Strategy')
        all_pis.append(pis)


simulation_data = {
        'all_counts': all_counts,
        'all_sums': all_sums,
        'all_pis': all_pis,
        }

print('Saving...')
with open(save_f+'/simulation_data.p', 'wb') as fp:
    pickle.dump(simulation_data, fp)

arm_counts = np.zeros(args.N)
for t in range(1, args.T+1):
    arm_counts += all_counts[0][t]

if args.save_rewards:
    with open(save_f+'/all_rewards.p', 'wb') as fp:
        pickle.dump(all_rewards, fp)


