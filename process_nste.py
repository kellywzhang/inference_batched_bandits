import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import os
import argparse
import json
import time
import pickle
from scipy.stats import chi2
import re

from utils import to_precision, dict_sum

parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, default='TS', \
        choices=['TS', 'epsilon', 'independent'])
parser.add_argument('--N', type=int, default=100000,
        help='Number of monte carlo simulations')
parser.add_argument('--n', type=int, default=100,
        help='Batch size')
parser.add_argument('--T', type=str, default='5,10,15,20,25',
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

# non-save arguments
parser.add_argument('--path', type=str, default='./simulations',
        help='Where to save results' )
parser.add_argument('--load_results', type=int, default=0,
        help='Only load results from a previous run of process.py')
parser.add_argument('--verbose', type=int, default=0,
        help='Prints more details')
parser.add_argument('--estvar', type=int, default=1,
        help='Estimate the variance')
parser.add_argument('--adjust', type=int, default=1,
        help='Use adjusted power to only allow feasible solutions (proper Type-1 error control)')
parser.add_argument('--awaipw', type=int, default=1,
        help='Use AW-AIPW estimator')
parser.add_argument('--Wdecorrelated', type=int, default=1,
        help='Use W-decorrealted estimator')
parser.add_argument('--bols_nste', type=int, default=1,
        help='Use BOLS NSTE estimator')
parser.add_argument('--nonstationary_path', type=str, default='./nonstationary_means',
        help='Path to folder with nonstationary mean files')
parser.add_argument('--null_means', type=str, default='0,0',
        help='Null expected rewards for each arm (file name in non-stationary case)')
parser.add_argument('--sparseT', type=str, default=None,
        help='Evaluate estimators not at every batch')

args = parser.parse_args()
print( vars(args) )
assert (args.K == 2)

plt.rcParams.update({'font.size': 15})

path = args.path
nonsave_args = ['path', 'load_results', 'estvar', 'adjust', 'awaipw', 'Wdecorrelated', \
        'bols_nste', 'verbose', 'nonstationary_path', 'null_means', 'sparseT']
save_args = [ '{}={}'.format(key, val) for key, val in vars(args).items() if key not in nonsave_args ]
save_str = '_'.join( save_args )
save_f_load = os.path.join( path, save_str)

if args.estvar: 
    all_save_f = os.path.join( path, save_str, 'estimate_variance')
else:
    all_save_f = os.path.join( path, save_str, 'known_variance')

if not os.path.isdir( save_f_load ):
    os.mkdir( save_f_load )
if not os.path.isdir( all_save_f ):
    os.mkdir( all_save_f )


alpha = 0.05
Tvals = [int(t) for t in args.T.split(',')]
save_f_list = []
for T in Tvals:
    arg_dict = vars(args)
    arg_dict['T'] = T
    means_f = args.means
    means_f = re.sub(r'_T\w+.txt', '_T{}.txt'.format(T), means_f)
    arg_dict['means'] = means_f
    save_args = [ '{}={}'.format(key, val) for key, val in arg_dict.items() if key not in nonsave_args ]
    save_str = '_'.join( save_args )
    save_f = os.path.join( path, save_str )
    save_f_list.append( save_f )


all_power_T = {}
all_se = []
all_nste = []
for i, T in enumerate(Tvals):
    print('\n', T)

    means_f = args.means
    means_f = re.sub(r'_T\w+.txt', '_T{}.txt'.format(T), means_f)
    with open( os.path.join(args.nonstationary_path, means_f), 'r' ) as f:
        lines = f.readlines()
    means0, means1 = lines[0], lines[1]
    true_means = [ [ float(x) for x in means0.split(',') ], [ float(x) for x in means1.split(',') ] ]
    assert( len(true_means[0]) == len(true_means[1] ) )
    assert( len(true_means[0]) == T )

    nste = np.array(true_means[1]) - np.array(true_means[0])
    all_nste.append( nste )
    print( "Difference in Arm Means", nste )

    if args.estvar:
        save_f = os.path.join( save_f_list[i], 'estimate_variance')
    else:
        save_f = os.path.join( save_f_list[i], 'known_variance')

    power_dict = pickle.load( open( os.path.join( save_f, \
            'power_dict_adjust={}.p'.format(args.adjust)), 'rb' ) )

    for key, val in power_dict.items():
        if key not in all_power_T.keys():
            all_power_T[key] = []
        all_power_T[key].append( val[T][alpha][0] )
        all_se.append( val[T][alpha][1] )

print("Maximum standard error: {}".format(np.max(all_se)))

strategy2name = {
    'TS': 'Thompson Sampling',
    'epsilon': r'$\epsilon$'+'-Greedy',
    'independent': 'Independently Sampled',
}

key2color = { 
    'ols': 'C0',
    'bols': 'C1',
    'Wdecorrelated': 'C2',
    'bols_nste': 'm', 
    'awaipw': 'C7',
}

key2name = {
    'ols': 'OLS',
    'bols': 'BOLS',
    'Wdecorrelated': 'W-Decorrelated',
    'bols_nste': 'BOLS NSTE',
    'awaipw': 'AW-AIPW', 
}

order_index = {
    'ols': 1,
    'Wdecorrelated': 3,
    'awaipw': 5,
    'bols': 8,
    'bols_nste': 14
}


# Prepare final results
plot_keys = ['ols', 'bols']
if args.Wdecorrelated:
    plot_keys.append('Wdecorrelated')
if args.awaipw:
    plot_keys.append('awaipw')
if args.bols_nste:
    plot_keys.append('bols_nste')

print('plot_keys', plot_keys)


# Power plots
plt.rcParams.update({'font.size': 15})
fig = plt.figure( figsize=(10,5) )

title_size = 18
label_size = 18
keys = [k for k in all_power_T.keys()]
keys.sort(key=lambda x: order_index[x] )
print(keys, plot_keys)

for key in keys:
    if key in plot_keys:
        plt.plot([t for t in range(5, args.T+5, 5)], all_power_T[key], label=key2name[key], color=key2color[key])
plt.xlabel('Batches (T)', fontsize=label_size)
plt.ylabel('Power', fontsize=label_size)
plt.ylim(bottom=0, top=1)
plt.xlim(left=0, right=args.T+1)

if args.estvar:
    estvarstr = "estimated variance"
else:
    estvarstr = "known variance"
plt.title("{} ({})".format( strategy2name[args.strategy], estvarstr ), fontsize=title_size)
plt.legend(fontsize='large', loc='right')
plt.savefig( os.path.join( all_save_f, 'power_plot_alpha={}.png'.format(alpha) ),
            bbox_inches='tight')
plt.close()


# Plotting Treatment Effect
styles=['-', '--', '-.', ':', (0, (5, 10))]

fig = plt.figure( figsize=(10,3) )
for i, t in enumerate(Tvals):
    all_nste[i] = [ x for x in all_nste[i] ] + [None]*(args.T-t)
    #all_nste[i] = all_nste[i] + [None]*(args.T-t)
    plt.plot([t for t in range(1,args.T+1)], [ x for x in all_nste[i] ], label='T={}'.format(t), \
            linestyle=styles[i], color='k')
plt.ylabel('Treatment Effect', fontsize=label_size)
plt.xlabel('Batches (T)', fontsize=label_size)
plt.xlim(left=0, right=args.T+1)
plt.ylim(bottom=-1.5, top=3)
plt.savefig( os.path.join( all_save_f, 'margins.png' ),
            bbox_inches='tight')
plt.close()


