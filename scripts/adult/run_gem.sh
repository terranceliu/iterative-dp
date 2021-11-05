#!/bin/bash

DATASET=adult
ADULT_SEED=0

MARGINAL=3
WORKLOAD=286
WORKLOAD_SEED=0
NUM_RUNS=5

MAX_ITERS=100
ALPHA=0.67
SYNDATA_SIZE=1000

LR=1e-4

for RUN in $(seq $NUM_RUNS)
do
  for EPSILON in 0.1 0.15 0.2 0.25 0.5 1.0
  do
    for T in 30 40 50 60 70 80 90 100 125 150 175 200
    do
      python gem.py --dataset $DATASET --verbose \
      --marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
      --epsilon $EPSILON --T $T \
      --lr $LR --max_iters $MAX_ITERS \
      --alpha $ALPHA --syndata_size $SYNDATA_SIZE \
       --adult_seed $ADULT_SEED

      rm -rf save/gem/${DATASET}_${ADULT_SEED}/${MARGINAL}_${WORKLOAD}_${WORKLOAD_SEED}/${EPSILON}_${T}_${ALPHA}_${SYNDATA_SIZE}
    done
  done
done