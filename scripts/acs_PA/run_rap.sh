#!/bin/bash

DATASET=acs_2018_1yr
STATE=PA

NUM_RUNS=5
MARGINAL=3
WORKLOAD=4096
WORKLOAD_SEED=0

N=1000

for K in 2 5 10 25 50
do
  for T in 2 5 10 15 20 25 50 75 100
  do
    for EPSILON in 0.1 0.15 0.2 0.25 0.5 1.0
    do
      python rap.py --dataset $DATASET --state $STATE --verbose \
      --num_runs $NUM_RUNS --marginal $MARGINAL \
      --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
      --epsilon $EPSILON --T $T --K $K --n $N \
      --lr 0.1
    done
  done
done
