#!/bin/bash

DATASET=adult
ADULT_SEED=0

NUM_RUNS=5
MARGINAL=3
WORKLOAD=286
WORKLOAD_SEED=0

for SAMPLES in 25 50 100 250 500
do
  for ETA in 1 2 3 4 5
  do
    python dualquery.py --dataset $DATASET --adult_seed $ADULT_SEED \
      --num_runs $NUM_RUNS --marginal $MARGINAL \
      --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
      --eps_ckpts 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.5 1.0 \
      --eta $ETA --samples $SAMPLES
  done
done