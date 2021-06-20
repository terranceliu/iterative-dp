#!/bin/bash

DATASET=adult-small
ADULT_SEED=0

NUM_RUNS=5
MARGINAL=3
WORKLOAD=35
WORKLOAD_SEED=0

MAXITERS=25

for EPSILON in 0.1 0.15 0.2 0.25 0.5 1.0
do
  for T in 20 30 40 50 75 100 125 150 175 200
  do
    python pep.py --dataset $DATASET --adult_seed $ADULT_SEED \
    --num_runs $NUM_RUNS --marginal $MARGINAL \
    --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
    --epsilon $EPSILON --T $T --iters $MAXITERS
  done
done