#!/bin/bash

DATASET=adult-small
ADULT_SEED=0

NUM_RUNS=1
MARGINAL=3
WORKLOAD=35
WORKLOAD_SEED=0

for T in 50 75 100 150 200 250 300
do
  for EPSILON in 1.0 0.5 0.25 0.2 0.15 0.1
  do
    python mwem.py --dataset $DATASET \
    --num_runs $NUM_RUNS --marginal $MARGINAL \
    --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
    --epsilon $EPSILON --T $T --adult_seed $ADULT_SEED
  done
done