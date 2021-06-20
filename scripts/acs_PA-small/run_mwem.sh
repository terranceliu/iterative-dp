#!/bin/bash

DATASET=acs_2018_1yr-small
STATE=PA

NUM_RUNS=5
MARGINAL=3
WORKLOAD=455
WORKLOAD_SEED=0

for T in 50 75 100 150 200 250 300
do
  for EPSILON in 1.0 0.5 0.25 0.2 0.15 0.1
  do
    python mwem.py --dataset $DATASET --state $STATE  \
    --num_runs $NUM_RUNS --marginal $MARGINAL \
    --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
    --epsilon $EPSILON --T $T
  done
done