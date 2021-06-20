#!/bin/bash

DATASET=acs_2018_1yr
STATE=PA

# PA-10
#DATASET_PUB=acs_2010_1yr
#STATE_PUB=$STATE

# OH-18
DATASET_PUB=$DATASET
STATE_PUB=OH

NUM_RUNS=5
MARGINAL=3
WORKLOAD=4096
WORKLOAD_SEED=0
FRAC=1.0
FRAC_SEED=0

MAX_ITERS=25

for EPSILON in 0.1 0.15 0.2 0.25 0.5 1.0
do
  for T in 20 40 60 80 100 120 140 160 180 200
  do
    python pep_pub.py --dataset $DATASET --state $STATE  \
    --dataset_pub $DATASET_PUB --state_pub $STATE_PUB\
    --num_runs $NUM_RUNS --marginal $MARGINAL \
    --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
    --pub_frac $FRAC --frac_seed $FRAC_SEED \
    --epsilon $EPSILON --T $T --iters $MAX_ITERS
  done
done