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

from utils import to_precision, dict_sum

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
    save_f = os.path.join( path, save_str, 'estimate_variance')
else:
    save_f = os.path.join( path, save_str, 'known_variance')

if not os.path.isdir( save_f ):
    os.mkdir( save_f )

prior_means = [ float(x) for x in args.prior_means.split(',') ]
alphas = [ 0.05 ]
if args.sparseT is None:
    Tvals = [t for t in range(1, args.T+1)]
else:
    Tvals = [ int(x) for x in args.sparseT.split(",") ]

if not args.nonstationary:
    true_means = [ float(x) for x in args.means.split(',') ]
    assert( len(true_means) == args.K )

    margin = true_means[1]-true_means[0]        # margin := Treatment effect
    null = margin == 0
    save_raw_null = os.path.join( path, save_str.replace("means={}".format(args.means), \
            "means={}".format(args.null_means)) )
    if args.estvar:
        save_f_null = os.path.join( save_raw_null, 'estimate_variance' )
    else:
        save_f_null = os.path.join( save_raw_null, 'known_variance' )
    print("Difference in Arm Means", margin)

else:
    with open( os.path.join(args.nonstationary_path, args.means), 'r' ) as f:
        lines = f.readlines()
    means0, means1 = lines[0], lines[1]
    true_means = [ [ float(x) for x in means0.split(',') ], [ float(x) for x in means1.split(',') ] ]
    assert( len(true_means[0]) == len(true_means[1] ) )
    assert( len(true_means[0]) == args.T )

    print( "Difference in Arm Means", np.array(true_means[1]) - np.array(true_means[0]) )
    null = np.equal(0, np.array(true_means[1]) - np.array(true_means[0])).all()
    if null:
        save_f_null = save_f
        save_raw_null = save_f_load
    else:
        save_raw_null = os.path.join( path, save_str.replace("means={}".format(args.means), \
                "means={}".format(args.null_means)) )
        if args.estvar:
            save_f_null = os.path.join( save_raw_null, 'estimate_variance' )
        else:
            save_f_null = os.path.join( save_raw_null, 'known_variance' )

    # Plot Treatment Effect
    fig = plt.figure( figsize=(10,3) )
    plt.plot( [t for t in range(1, args.T+1)], np.array(true_means[1]) - np.array(true_means[0]), color='k',
            label='Treatment Effect')
    plt.xlabel('Batches (T)', fontsize=15)
    plt.legend(fontsize='large')
    plt.ylim(bottom=-2, top=2)
    plt.savefig( os.path.join( save_f, 'treatment_effect.png' ),
            bbox_inches='tight')
    plt.close()

    # Plot Arm Means
    fig = plt.figure( figsize=(10,3) )
    plt.plot( [t for t in range(1, args.T+1)], np.array(true_means[1]), color='k', label='Arm 1', linestyle='--')
    plt.plot( [t for t in range(1, args.T+1)], np.array(true_means[0]), color='k', label='Arm 0')
    plt.xlabel('Batches (T)', fontsize=15)
    plt.ylabel('Expected Reward', fontsize=18)
    plt.ylim(bottom=-2, top=2)
    plt.legend(fontsize='large', loc='lower left')
    plt.savefig( os.path.join( save_f, 'arm_means.png' ),
            bbox_inches='tight')
    plt.close()

if null and args.adjust:
    if not os.path.isdir( os.path.join( save_f, 'cutoff_adjustments' ) ):
        os.mkdir( os.path.join( save_f, 'cutoff_adjustments' ) )

strategy2name = {
            'TS': 'Thompson Sampling',
            'epsilon': r'$\epsilon$'+'-Greedy',
            'independent': 'Independently Sampled',
        }


def make_hist(name, vals, plot_normal=(0,1), density=True, power=False, title_size=13, hist_type='errors'):
    if density:
        vals = vals.copy() / np.sqrt( args.var )
    mu = np.mean(vals)
    var = np.var( vals, ddof=1 )
    title_string = "Distribution under {}".format( strategy2name[args.strategy] ) 
       
    plt.hist( vals, bins=100, density=density, label='Empirical Distribution', color='C2')
    
    if power:
        alpha = 0.05
        cutoff = math.fabs( stats.norm.ppf( alpha / 2 ) )
        plt.axvline( x= cutoff, color='k', label='Normal Cutoffs (\u03B1=0.05)' )
        plt.axvline( x= -cutoff, color='k' )

        power = np.greater( np.abs( vals )/ math.sqrt(var), cutoff ).mean()
        power = np.round(power*100,1)

    plt.legend(loc='center right', fontsize='small',
                bbox_to_anchor=(1.35, 0.5))
    if hist_type=='errors':
        if args.strategy == 'epsilon':
            plt.text(-4.3, 0.2, "Type-1 error:\n{} %".format( power ), fontsize=13)
        else:
            plt.text(-4.5, 0.2, "Type-1 error:\n{} %".format( power ), fontsize=13)
    plt.title( title_string , fontsize=title_size )
    plt.savefig( os.path.join( save_f, name ) + '.png', bbox_inches='tight' )
    plt.close()


def print_results(t, alphas, print_dict):
    print( "-----------------------\nt={}".format(t) )
    for alpha in alphas:
        print( 'alpha={}'.format(alpha))
        if null:
            print('Type-1 Error={}'.format(print_dict[t][alpha]) )
        else:
            print('Power={}'.format(print_dict[t][alpha]) )


def calculate_power(alphas, cutoffs, estimates, save_dict):
    for A, cutoff in zip(alphas, cutoffs):
        vals = np.abs( estimates ) 
        ht = np.greater( vals, cutoff )
        power = ht.mean()
        se = np.std( ht, ddof=1) / math.sqrt(args.N)
        save_dict[A] = (power, se)
    return save_dict


def calculate_cutoff_adjustment(alphas, vals, orig_cutoffs=None):
    cutoffs = {}
    for k, A in enumerate(alphas):
        adjusted_cutoff = np.quantile( np.abs(vals), 1-A )
        if orig_cutoffs is not None:
            # We do not adjust the cutoff if using the original cutoff does not inflate the Type-1 error
            cutoffs[A] = max( adjusted_cutoff, orig_cutoffs[k] )
        else:
            cutoffs[A] = adjusted_cutoff
    return cutoffs


def ols_inference(simulation_dict, alphas, power_dict):
    power_dict['ols'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    
    all_sums = simulation_dict['all_sums']
    all_counts = simulation_dict['all_counts']
    if args.estvar:
        all_rewards_array = simulation_dict['all_rewards_array']
        mask0 = simulation_dict['mask0']

    if args.adjust:
        if null:
            adjusted_cutoffs = {}
        else:
            with open( os.path.join( save_f_null, 'cutoff_adjustments', 'ols.json' ), 'r' ) as f:
                adjusted_cutoffs = json.load( f )

    print( '\nOLS' )
    for t in Tvals:
        sums0 = dict_sum(all_sums[0], t)
        counts0 = dict_sum(all_counts[0], t)
        sums1 = dict_sum(all_sums[1], t)
        counts1 = dict_sum(all_counts[1], t)

        ols1_est = np.divide( sums1, counts1, out=np.zeros_like(sums1), where=counts1!=0 )
        ols0_est = np.divide( sums0, counts0, out=np.zeros_like(sums0), where=counts0!=0 )
        ols_margin_est = ols1_est - ols0_est

        if args.estvar:
            all_residuals = []
            for k in range(1,t+1):
                residuals_k = all_rewards_array[k-1] - mask0[k-1] * np.expand_dims(ols0_est,1) \
                        - (1-mask0[k-1]) * np.expand_dims(ols1_est,1)
                all_residuals.append(residuals_k)
            all_residuals = np.concatenate( all_residuals, 1 )
            noise_std = np.std(all_residuals, ddof=2, axis=1)
        else:
            noise_std = np.ones(args.N)*math.sqrt(args.var)

        ols_margin_stat = np.sqrt( counts0 * counts1 / (counts0 + counts1) ) * ( ols_margin_est / noise_std )
        cutoffs = [ math.fabs( scipy.stats.norm.ppf( alpha / 2 ) ) for alpha in alphas ]
        
        if args.adjust:
            if null:
                adjusted_cutoffs[t] = calculate_cutoff_adjustment(alphas, ols_margin_stat, \
                        orig_cutoffs=cutoffs)
            else:
                # get adjusted cutoffs
                cutoffs = [ v for k, v in adjusted_cutoffs[str(t)].items() if float(k) in alphas ]
        
        calculate_power(alphas, cutoffs, ols_margin_stat, power_dict['ols'][t])
        
        if args.T <= 5 or (args.T > 5 and t % 5 == 0):
            make_hist( 'ols_distribution_t={}'.format(t), ols_margin_stat, power=True )

        if t % 5 == 0:
            print_results(t, alphas, print_dict=power_dict['ols'])
        
    if args.adjust:
        if null:
            with open( os.path.join( save_f, 'cutoff_adjustments', 'ols.json' ), 'w' ) as f:
                json.dump( adjusted_cutoffs, f, indent=4 )

    return noise_std


def bols_inference(simulation_dict, alphas, power_dict, nste=False):
    # Find cutoff values using Student-t distribution
    if args.estvar:
        est_cutoffs = {}
        t_sample = np.array( [ np.random.standard_t(df=args.n-2, size=100*args.N) for x in range(args.T) ] )
        for t in Tvals:
            est_cutoffs[t] = []
            if not nste:
                # Stationary treatment effect
                for alpha in alphas:
                    # Simulate cutoffs using Student-t distribution
                    ave_t_sample = np.sum( t_sample[:t], axis=0 ) / math.sqrt(t)
                    cutoff = stats.mstats.mquantiles( ave_t_sample, prob=1-alpha/2 )[0]
                    est_cutoffs[t].append(cutoff)
                    se = np.std( ave_t_sample > cutoff ) / args.N
                    if args.verbose:
                        print('BOLS cutoff', 't={}'.format(t), 'alpha={}'.format(alpha), cutoff, se)
            else:
                # Non-stationary treatment effect
                for alpha in alphas:
                    # Simulate cutoffs using Student-t distribution
                    squared_t_sample = np.sum( np.square(t_sample[:t]), axis=0 )
                    cutoff = stats.mstats.mquantiles( squared_t_sample, prob=1-alpha )[0]
                    est_cutoffs[t].append(cutoff)
                    se = np.std( squared_t_sample > cutoff ) / args.N
                    if args.verbose:
                        print('BOLS NSTE cutoff', 't={}'.format(t), 'alpha={}'.format(alpha), cutoff, se)
    
    if nste:
        power_dict['bols_nste'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    else:
        power_dict['bols'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    bols_dict = { 'est0': [], 'est1': [], 'stats': [] }
    
    all_sums = simulation_dict['all_sums']
    all_counts = simulation_dict['all_counts']
    if args.estvar:
        all_rewards_array = simulation_dict['all_rewards_array']
        mask0 = simulation_dict['mask0']
   
    if nste:
        print( '\nBOLS NSTE' )
    else:
        print( '\nBOLS' )
    for t in range(1, args.T+1):
        sums0 = dict_sum(all_sums[0], t)
        counts0 = dict_sum(all_counts[0], t)
        sums1 = dict_sum(all_sums[1], t)
        counts1 = dict_sum(all_counts[1], t)

        ols1_est = np.divide( sums1, counts1, out=np.zeros_like(sums1), where=counts1!=0 )
        ols0_est = np.divide( sums0, counts0, out=np.zeros_like(sums0), where=counts0!=0 )
        ols_margin_est = ols1_est - ols0_est

        bols_t_est0 = np.divide( all_sums[0][t], all_counts[0][t] )
        bols_t_est1 = np.divide( all_sums[1][t], all_counts[1][t] )
        bols_dict['est0'].append( bols_t_est0 )
        bols_dict['est1'].append( bols_t_est1 )

        bols_t_est = bols_t_est1 - bols_t_est0
        bols_t_sd = 1/np.sqrt( all_counts[0][t] * all_counts[1][t] / ( all_counts[0][t] + all_counts[1][t] ) )
        bols_t_stat = bols_t_est / bols_t_sd
        bols_dict['stats'].append( bols_t_stat )

    for t in Tvals:
        # Estimate variance
        if args.estvar:
            all_batch_var = []
            for b in range(1,t+1):
                residuals = all_rewards_array[b-1] \
                        - mask0[b-1] * np.expand_dims( bols_dict['est0'][b-1], 1 ) \
                        - (1-mask0[b-1]) * np.expand_dims( bols_dict['est1'][b-1], 1 )
                batch_var = np.var(residuals, axis=1, ddof=2)
                all_batch_var.append( batch_var )
            noise_std = np.sqrt( np.concatenate( [np.expand_dims(x,0) for x in all_batch_var], axis=0) )
        else:
            noise_std = np.ones(args.N)*math.sqrt(args.var)
        
        # Simulate cutoffs
        if args.estvar:
            cutoffs = est_cutoffs[t]
        else:
            # variance known
            if not nste:
                # Stationary treatment effect
                # Cutoffs from Normal distribution
                cutoffs = [ math.fabs( scipy.stats.norm.ppf( alpha / 2 ) ) for alpha in alphas ]
            else:
                # Non-stationary treatment effect
                cutoffs = [ stats.chi2.ppf(1-alpha, df=t) for alpha in alphas ]
   
        if not nste:
            # Stationary treatment effect
            bols_stat = np.sum( np.array( bols_dict['stats'][:t] ) / noise_std, axis=0) / math.sqrt(t)
            calculate_power(alphas, cutoffs, bols_stat, power_dict['bols'][t])
        
            if args.T <= 5 or (args.T > 5 and t % 5 == 0):
                make_hist( 'bols_distribution_t={}'.format(t), bols_stat, power=True )

            if t % 5 == 0:
                print_results(t, alphas, print_dict=power_dict['bols'])
        else:
            # Non-stationary treatment effect
            bols_nste_stat = np.sum( np.square( np.array( bols_dict['stats'][:t] ) / noise_std ), axis=0)
            calculate_power(alphas, cutoffs, bols_nste_stat, power_dict['bols_nste'][t])
            if t % 5 == 0:
                print_results(t, alphas, print_dict=power_dict['bols_nste'])

    return bols_dict


def awaipw_inference(simulation_dict, alphas, power_dict):
    power_dict['awaipw'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    
    all_sums = simulation_dict['all_sums']
    all_counts = simulation_dict['all_counts']
    all_rewards = simulation_dict['all_rewards']

    all_mu1 = { 0: np.zeros(args.N) }
    all_mu0 = { 0: np.zeros(args.N) }
    
    if args.adjust:
        if null:
            adjusted_cutoffs = {}
        else:
            with open( os.path.join( save_f_null, 'cutoff_adjustments', 'awaipw.json' ), 'r' ) as f:
                adjusted_cutoffs = json.load( f )
    
    for t in range(1, args.T+1):
        # Update model mu
        sums0 = dict_sum(all_sums[0], t)
        counts0 = dict_sum(all_counts[0], t)
        sums1 = dict_sum(all_sums[1], t)
        counts1 = dict_sum(all_counts[1], t)

        all_mu1[t] = sums1 / counts1
        all_mu0[t] = sums0 / counts0


    print( '\nAW-AIPW' )
    for t in Tvals:
        # weights: h_t = sqrt(pi); mu_hat is sample mean
        hsum1 = 0; hsum0 = 0; Q1 = 0; Q0 = 0

        # First, we calculate the estimators Q1, Q0
        for k in range(1,t+1):
            # IPW portion
            ipw1 = all_sums[1][k] / all_pis[k-1]
            ipw0 = all_sums[0][k] / (1-all_pis[k-1])

            # Augmented model portion
            aug1 = all_mu1[k-1] * ( ( 1 - 1 / all_pis[k-1] ) * all_counts[1][k] + all_counts[0][k] )
            aug0 = all_mu0[k-1] * ( ( 1 - 1 / (1-all_pis[k-1]) ) * all_counts[0][k] + all_counts[1][k] )

            ht1 = np.sqrt( all_pis[k-1] )
            ht0 = np.sqrt( 1-all_pis[k-1] )

            Q1 = Q1 + ht1 * ( ipw1 + aug1 )
            Q0 = Q0 + ht0 * ( ipw0 + aug0 )

            hsum1 += args.n * ht1
            hsum0 += args.n * ht0

        Q1 = Q1 / hsum1
        Q0 = Q0 / hsum0
        
        v1_num = 0; v0_num = 0; cov_num = 0
        for k in range(1, t+1):
            all_err0 = []; all_err1 = []; all_cov = []
            for i in range(args.N):
                ipw_err0 = ( all_rewards[0][k][i] - Q0[i] ) / (1-all_pis[k-1][i])
                aug_err0 = ( 1 - 1/(1-all_pis[k-1][i]) ) * ( all_mu0[k-1][i] - Q0[i] )
                augonly_err0 = np.array( all_counts[1][k][i] * [ all_mu0[k-1][i] - Q0[i] ] )
                err0 = np.hstack( [ ipw_err0 + aug_err0, augonly_err0 ] )
                all_err0.append( sum( np.square( err0 ) ) )

                ipw_err1 = ( all_rewards[1][k][i] - Q1[i] ) / all_pis[k-1][i]
                aug_err1 = ( 1 - 1/all_pis[k-1][i] ) * ( all_mu1[k-1][i] - Q1[i] )
                augonly_err1 = np.array( all_counts[0][k][i] * [ all_mu1[k-1][i] - Q1[i] ] )
                err1 = np.hstack( [ augonly_err1, ipw_err1 + aug_err1 ] )
                all_err1.append( sum( np.square( err1 ) ) )

                all_cov.append( sum( err1 * err0 ) )

            ht1_square = all_pis[k-1]
            ht0_square = 1-all_pis[k-1]

            v1_num += ht1_square * np.array(all_err1)
            v0_num += ht0_square * np.array(all_err0)
            cov_num += np.sqrt( ht1_square ) * np.sqrt( ht0_square ) * np.array(all_cov)

        v1 = v1_num / np.square( hsum1 )
        v0 = v0_num / np.square( hsum0 )
        cov = cov_num / ( hsum1 * hsum0 )
        
        # Calculate test statistic
        awaipw_stat = ( Q1 - Q0 ) / np.sqrt( v1 + v0 -2 * cov )
        cutoffs = [ math.fabs( scipy.stats.norm.ppf( alpha / 2 ) ) for alpha in alphas ]
        
        if args.adjust:
            if null:
                adjusted_cutoffs[t] = calculate_cutoff_adjustment(alphas, awaipw_stat, \
                        orig_cutoffs=cutoffs)
            else:
                # get adjusted cutoffs
                cutoffs = [ v for k, v in adjusted_cutoffs[str(t)].items() if float(k) in alphas ]
        
        calculate_power(alphas, cutoffs, awaipw_stat, power_dict['awaipw'][t])
        print_results(t, alphas, print_dict=power_dict['awaipw'])
    
    if args.adjust:
        if null:
            with open( os.path.join( save_f, 'cutoff_adjustments', 'awaipw.json' ), 'w' ) as f:
                json.dump( adjusted_cutoffs, f, indent=4 )


def Wdecorrelated_inference(simulation_dict, alphas, power_dict, noise_std):
    power_dict['Wdecorrelated'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    
    all_sums = simulation_dict['all_sums']
    all_counts = simulation_dict['all_counts']
    all_rewards = simulation_dict['all_rewards']
   
    if not null:
        with open( os.path.join( save_f_null, 'lambda.json' ), 'r') as f:
            null_lambdas = json.load(f)
    else:
        null_lambdas = {}
    
    if args.adjust:
        if null:
            adjusted_cutoffs = {}
        else:
            with open( os.path.join( save_f_null, 'cutoff_adjustments', 'Wdecorrelated.json' ), 'r' ) as f:
                adjusted_cutoffs = json.load( f )
    
    print( '\nW-decorrelated' )
    for t in Tvals:
        sums0 = dict_sum(all_sums[0], t)
        counts0 = dict_sum(all_counts[0], t)
        sums1 = dict_sum(all_sums[1], t)
        counts1 = dict_sum(all_counts[1], t)
        
        if null:
            lam = np.quantile( np.minimum( counts0, counts1 ) / np.log( args.n*t ), 1/(args.n*t) )
            null_lambdas[t] = lam
        else:
            lam = null_lambdas[str(t)]

        mean0 = np.divide( sums0, counts0 )
        mean1 = np.divide( sums1, counts1 )

        R = 1/(1+lam)
        Rvec = np.array([ R*(1-R)**j for j in range(args.n * t) ])

        residual0 = [ np.concatenate( [ all_rewards[0][k][i] - mean0[i] for k in range(1,t+1) ] ) \
                for i in range(args.N) ]
        residual1 = [ np.concatenate( [ all_rewards[1][k][i] - mean1[i] for k in range(1,t+1) ] ) \
                for i in range(args.N) ]
        all_residuals0_padded = \
                np.array( [ np.concatenate( [ residual0[i], np.zeros( args.n*t - len(residual0[i] ) ) ] ) \
                for i in range(args.N) ] )
        all_residuals1_padded = \
                np.array( [ np.concatenate( [ residual1[i], np.zeros( args.n*t - len(residual1[i] ) ) ] ) \
                for i in range(args.N) ] )

        arm0_correction = np.sum( np.multiply( Rvec, all_residuals0_padded ), axis=1 )
        arm1_correction = np.sum( np.multiply( Rvec, all_residuals1_padded ), axis=1 )

        RvecSquare = np.square( [ R*(1-R)**j for j in range(args.n * args.T) ] )
        arm0_var = np.array( [ np.sum( RvecSquare[:counts0[i]] ) for i in range(args.N) ] )
        arm1_var = np.array( [ np.sum( RvecSquare[:counts1[i]] ) for i in range(args.N) ] )

        W0_est = mean0 + arm0_correction
        W1_est = mean1 + arm1_correction

        W_stat = ( W1_est - W0_est ) / np.sqrt( arm0_var + arm1_var )
        W_stat = W_stat / noise_std

        if args.T <= 5 or (args.T > 5 and t % 5 == 0):
            make_hist( 'Wdecorrelated_distribution_t={}'.format(t), W_stat, power=True )

        cutoffs = [ math.fabs( scipy.stats.norm.ppf( alpha / 2 ) ) for alpha in alphas ]
        
        if args.adjust:
            if null:
                adjusted_cutoffs[t] = calculate_cutoff_adjustment(alphas, W_stat, \
                        orig_cutoffs=cutoffs)
            else:
                # get adjusted cutoffs
                cutoffs = [ v for k, v in adjusted_cutoffs[str(t)].items() if float(k) in alphas ]
        
        calculate_power(alphas, cutoffs, W_stat, power_dict['Wdecorrelated'][t])
        print_results(t, alphas, print_dict=power_dict['Wdecorrelated'])
        
    with open( os.path.join( save_f, 'lambda.json' ), 'w' ) as f:
        json.dump(null_lambdas, f, indent=4)
    
    if args.adjust:
        if null:
            with open( os.path.join( save_f, 'cutoff_adjustments', 'Wdecorrelated.json' ), 'w' ) as f:
                json.dump( adjusted_cutoffs, f, indent=4 )


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

def power_plots(plot_keys, power_dict, alphas):
    # Power plots
    title_size = 18
    label_size = 18
    
    keys = [k for k in power_dict.keys()]
    keys.sort(key=lambda x: order_index[x] )

    for alpha in alphas:
        fig = plt.figure( figsize=(10,5) )
        all_se = []
        for key in keys:
            if key in plot_keys:
                vals, se = zip(* [ power_dict[key][t][alpha] for t in Tvals] )
                plt.plot(Tvals, vals, label=key2name[key], color=key2color[key])
                all_se.append( np.max( se ) )
        plt.xlabel('Batches (T)', fontsize=label_size)
        if null:
            plt.ylabel('Type-1 Error', fontsize=label_size)
            plt.hlines( alpha, 1, args.T, label='Nominal '+r'$\alpha$', color='k' )
            if not args.nonstationary:
                plt.ylim(top=0.07)
        else:
            plt.ylabel('Power', fontsize=label_size)
            plt.ylim(bottom=0, top=1)
        if args.estvar:
            estvarstr = "estimated variance"
        else:
            estvarstr = "known variance"
        plt.title("{} ({})".format( strategy2name[args.strategy], estvarstr ), fontsize=title_size)
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.legend(fontsize='large')
        if null:
            plt.savefig( os.path.join( save_f, 'type1error_alpha={}.png'.format(alpha) ),
                    bbox_inches='tight')
        else:
            plt.savefig( os.path.join( save_f, 'power_alpha={}_adjusted={}.png'.format(alpha, args.adjust) ),
                    bbox_inches='tight')
        plt.close()

        return np.max(all_se)



########################
# Load simulation data #
########################
if not args.load_results:
    print('Loading...')
    with open(save_f_load+'/simulation_data.p', 'rb') as fp:
        simulation_data = pickle.load(fp)

    all_counts = simulation_data[ 'all_counts' ]
    all_sums = simulation_data[ 'all_sums' ]
    all_pis = simulation_data[ 'all_pis' ]

    if args.estvar or args.awaipw or args.Wdecorrelated:
        with open(save_f_load+'/all_rewards.p', 'rb') as fp:
            all_rewards = pickle.load(fp)

        if args.estvar:
            all_rewards_array = []
            mask0 = []
            for t in range(1, args.T+1):
                # Mask: 1 if arm 0 is sampled; 0 if arm 1 is sampled
                mask0.append( np.array( [ np.array( [1]*len(all_rewards[0][t][i]) + \
                        [0]*(args.n-len(all_rewards[0][t][i])) ) for i in range(args.N) ] ) )
                # All rewards
                rewards_t = np.array( [ np.concatenate( [ all_rewards[0][t][i], all_rewards[1][t][i] ] ) for i in range(args.N) ] )
                all_rewards_array.append( rewards_t )
    
    print('Done loading!')

    # Plot sampling probabilities
    for t in range(args.T):
        if args.T <= 5 or t % 5 == 0:
            make_hist('sampling_probabilities_t={}'.format(t+1), all_pis[t], plot_normal=False, 
                      density=False, hist_type='pis')

    # Prepare dictionaries
    power_dict = {}
    simulation_dict = {
        'all_sums': all_sums,
        'all_counts': all_counts,
    }
    if args.estvar:
        simulation_dict['all_rewards_array'] = all_rewards_array
        simulation_dict['mask0'] = mask0
    if args.awaipw or args.Wdecorrelated:
        simulation_dict['all_rewards'] = all_rewards

    # Perform inference
    ols_noise_std = ols_inference(simulation_dict, alphas, power_dict)
    bols_dict = bols_inference(simulation_dict, alphas, power_dict)
    if args.bols_nste:
        bols_nste_dict = bols_inference(simulation_dict, alphas, power_dict, nste=True)
    if args.Wdecorrelated:
        Wdecorrelated_dict = Wdecorrelated_inference(simulation_dict, alphas, power_dict, noise_std=ols_noise_std)
    if args.awaipw:
        awaipw_dict = awaipw_inference(simulation_dict, alphas, power_dict)
    pickle.dump( power_dict, open( os.path.join( save_f, \
            'power_dict_adjust={}.p'.format(args.adjust)), 'wb' ) )
else:
    power_dict = pickle.load( open( os.path.join( save_f, \
            'power_dict_adjust={}.p'.format(args.adjust)), 'rb' ) )

# Prepare final results
plot_keys = ['ols', 'bols']
if args.Wdecorrelated:
    plot_keys.append('Wdecorrelated')
if args.awaipw:
    plot_keys.append('awaipw')
if args.bols_nste:
    plot_keys.append('bols_nste')

print('plot_keys', plot_keys)

# Print final batch type-1 error / power
for alpha in alphas:
    print('\n\n\n=======================================================')
    if null:
        print("Batch {}, Type-1 Error:".format(args.T))
    else:
        print("Batch {}, Power:".format(args.T))
    print('alpha={}'.format(alpha))
    print('=======================================================')
    for key in plot_keys:
        print(key, [ power_dict[key][t][alpha][0] for t in Tvals])
    print('=======================================================')


# Plot results
if len(Tvals) > 2:
    max_se = power_plots(plot_keys, power_dict, alphas)
    print("Maximum standard error: {}".format(max_se))



