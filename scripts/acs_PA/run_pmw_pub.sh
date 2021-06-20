#!/bin/bash

DATASET=acs_2018_1yr
STATE=PA

# PA-10
DATASET_PUB=acs_2010_1yr
STATE_PUB=$STATE

# OH-18
#DATASET_PUB=$DATASET
#STATE_PUB=OH

NUM_RUNS=5
MARGINAL=3
WORKLOAD=4096
WORKLOAD_SEED=0
FRAC=1.0
FRAC_SEED=0

for EPSILON in 1.0 0.5 0.25 0.2 0.15 0.1
do
  for T in 50 75 100 150 200 250 300
  do
    python pmw_pub.py --dataset $DATASET --state $STATE  \
    --dataset_pub $DATASET_PUB --state_pub $STATE_PUB\
    --num_runs $NUM_RUNS --marginal $MARGINAL \
    --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
    --pub_frac $FRAC --frac_seed $FRAC_SEED \
    --epsilon $EPSILON --T $T
  done
done