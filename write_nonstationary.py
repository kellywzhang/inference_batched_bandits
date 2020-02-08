import numpy as np
import math
import os

means_path = './nonstationary_means'

if not os.path.isdir(means_path):
    os.mkdir(means_path)

##########################
# Nonstationary baseline #
##########################
for t in [5, 25]:
    vals = [ 1.5/t**2 * (x-t)**2 - 0.5 for x in range(t) ]
    baseline = [np.round(x,3) for x in vals]

    with open( os.path.join(means_path, 'baseline_025_margin_T{}.txt'.format(t) ), 'w') as f:
        f.write(','.join( [str(x) for x in baseline] ))
        f.write('\n')
        f.write(','.join([ str(np.round(baseline[x]+0.25,3)) for x in range(t) ]))
        f.write('\n')

    with open( os.path.join(means_path, 'baseline_zero_margin_T{}.txt'.format(t) ), 'w') as f:
        f.write(','.join( [str(x) for x in baseline] ))
        f.write('\n')
        f.write(','.join( [str(x) for x in baseline] ))
        f.write('\n')


##################################
# Nonstationary treatment effect #
##################################
for t in range(5,30,5):
    #####################
    # Sin shaped margin #
    #####################
    vals = [ np.sin(x*2*np.pi/t) for x in range(t) ]
    margins = [str(np.round(x,3)) for x in vals]
   
    try:
        assert len(margins) == t
    except:
        import ipdb; ipdb.set_trace()
        
    assert len(margins) == t

    with open(os.path.join(means_path, 'sin_margin_T{}.txt'.format(t)), 'w') as f:
        f.write(','.join([ '0.0' for x in range(t) ]))
        f.write('\n')
        f.write(','.join(margins))
        f.write('\n')

    ####################
    # Quadratic margin #
    ####################
    vals = [ 1.5/t**2 * (x-t)**2 - 0.5 for x in range(t) ]
    margins = [str(np.round(x,3)) for x in vals]

    with open(os.path.join(means_path, 'quadratic_margin_T{}.txt'.format(t)), 'w') as f:
        f.write(','.join([ '0.0' for x in range(t) ]))
        f.write('\n')
        f.write(','.join(margins))
        f.write('\n')

    ###############
    # Zero margin #
    ###############
    with open(os.path.join(means_path, 'zero_margin_T{}.txt'.format(t)), 'w') as f:
        f.write(','.join([ '0.0' for x in range(t) ]))
        f.write('\n')
        f.write(','.join([ '0.0' for x in range(t) ]))
        f.write('\n')

