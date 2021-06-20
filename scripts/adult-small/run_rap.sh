#!/bin/bash

DATASET=adult-small
ADULT_SEED=0

NUM_RUNS=5
MARGINAL=3
WORKLOAD=35
WORKLOAD_SEED=0

N=1000

for T in 2 5 10 25 50 75 100
do
  for K in 5 10 25 50 100
  do
    for EPSILON in 0.1 0.15 0.2 0.25 0.5 1.0
    do
      python rap.py --dataset $DATASET --verbose \
      --num_runs $NUM_RUNS --marginal $MARGINAL \
      --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
      --epsilon $EPSILON --T $T --K $K --adult_seed $ADULT_SEED \
      --lr 5e-3
    done
  done
done