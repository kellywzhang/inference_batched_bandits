
######################
# Stationary Setting #
######################
T=25
batch_size=25
clipping=0.1
allT="1"
for t in $(seq 5 5 25);
do
    allT="$allT,$t"
done

# Run stationary simulations
python run_simulation.py --strategy TS --n $batch_size --T $T --means 0,0 --clipping $clipping
python run_simulation.py --strategy TS --n $batch_size --T $T --means 0.25,0 --clipping $clipping

# Process stationary simulations
python process.py --strategy TS --n $batch_size --T $T --means 0,0 --clipping $clipping --verbose 1 --sparseT $allT
python process.py --strategy TS --n $batch_size --T $T --means 0.25,0 --clipping $clipping --verbose 1 --null_means 0,0 --sparseT $allT

###########################
# Non-Stationary Baseline #
###########################

# Make non-stationary arm mean files
python write_nonstationary.py

# Run non-stationary baseline reward simulations
python run_simulation.py --strategy TS --n $batch_size --T 25 --clipping $clipping --means baseline_zero_margin_T25.txt --nonstationary 1
python run_simulation.py --strategy TS --n $batch_size --T 25 --clipping $clipping --means baseline_025_margin_T25.txt --nonstationary 1

# Process non-stationary baseline reward simulations
python process.py --strategy TS --n $batch_size --T 25 --clipping $clipping --means baseline_zero_margin_T25.txt --nonstationary 1 --sparseT $allT
python process.py --strategy TS --n $batch_size --T 25 --clipping $clipping --means baseline_025_margin_T25.txt --nonstationary 1 --null_means baseline_zero_margin_T25.txt --sparseT $allT


###################################
# Non-Stationary Treatment Effect #
###################################

# Run non-stationary treatment simulations
for T in $(seq 5 5 25);
do
    python run_simulation.py --strategy TS --n $batch_size --T $T --clipping $clipping --means "zero_margin_T$T.txt" --nonstationary 1
    python run_simulation.py --strategy TS --n $batch_size --T $T --clipping $clipping --means "quadratic_margin_T$T.txt" --nonstationary 1
done

# Process non-stationary treatment simulations
for T in $(seq 5 5 25);
do
    python process.py --strategy TS --n $batch_size --T $T --clipping $clipping --means "zero_margin_T$T.txt" --nonstationary 1 --sparseT "$T"
    python process.py --strategy TS --n $batch_size --T $T --clipping $clipping --means "quadratic_margin_T$T.txt" --nonstationary 1 --null_means "zero_margin_T$T.txt" --sparseT "$T"
done

# Make nonstationary plots
allT="5"
for T in $(seq 10 5 25);
do
    allT="$allT,$T"
done

python process_nste.py --strategy TS --n $batch_size --T $allT --clipping $clipping --means "quadratic_margin_T$T.txt" --nonstationary 1 --null_means "zero_margin_T$T.txt" 

