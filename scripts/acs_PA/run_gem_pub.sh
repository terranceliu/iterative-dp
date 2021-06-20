#!/bin/bash

DATASET=acs_2018_1yr
STATE=PA

DATASET_PUB=acs_2010_1yr
STATE_PUB=PA
#DATASET_PUB=acs_2018_1yr
#STATE_PUB=CA
#DATASET_PUB=acs_2018_1yr
#STATE_PUB=OH

MARGINAL=3
WORKLOAD=4096
WORKLOAD_SEED=0
NUM_RUNS=5

MAX_ITERS=100
ALPHA=0.67
SYNDATA_SIZE=1000

LR=1e-4

for RUN in $(seq $NUM_RUNS)
do
  for EPSILON in 0.1 0.15 0.2 0.25
  do
    for T in 30 40 50 75 100 150 200 300 400 500
    do
      python gem.py --dataset $DATASET --state $STATE --verbose \
      --dataset_pub $DATASET_PUB --state_pub $STATE_PUB \
      --marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
      --epsilon $EPSILON --T $T \
      --lr $LR --max_iters $MAX_ITERS \
      --alpha $ALPHA --syndata_size $SYNDATA_SIZE

#      rm -rf save/gem_pub/${DATASET}_${STATE}/${MARGINAL}_${WORKLOAD}_${WORKLOAD_SEED}/${DATASET_PUB}_${STATE_PUB}/${EPSILON}_${T}_${ALPHA}_${SYNDATA_SIZE}
    done
  done
done