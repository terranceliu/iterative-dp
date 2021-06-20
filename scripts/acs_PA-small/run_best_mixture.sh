#!/bin/bash

DATASET=acs_2018_1yr-small
STATE=PA

# PA-10
#DATASET_PUB=acs_2010_1yr-small
#STATE_PUB=$STATE

# OH-18
DATASET_PUB=$DATASET
STATE_PUB=OH

NUM_RUNS=5
MARGINAL=5
WORKLOAD=3003
WORKLOAD_SEED=0
FRAC=1.0
FRAC_SEED=0

python best_mixture_error_pub.py --dataset $DATASET --state $STATE \
--dataset_pub $DATASET_PUB --state_pub $STATE_PUB \
--num_runs $NUM_RUNS --marginal $MARGINAL \
--workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--pub_frac $FRAC --frac_seed $FRAC_SEED