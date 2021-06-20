#!/bin/bash

DATASET=adult
ADULT_SEED=0

MARGINAL=3
WORKLOAD=286
WORKLOAD_SEED=0

MAX_IDXS=100
MAX_ITERS=100
SYNDATA_SIZE=1000

LR=1e-4

python gem_nondp.py --dataset $DATASET --adult_seed $ADULT_SEED \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--lr $LR --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--syndata_size $SYNDATA_SIZE --verbose --overwrite --reduce_attr

python gem_nondp.py --dataset $DATASET --adult_seed $ADULT_SEED \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--lr $LR --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--syndata_size $SYNDATA_SIZE --verbose --overwrite