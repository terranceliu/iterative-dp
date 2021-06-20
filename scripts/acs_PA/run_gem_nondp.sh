#!/bin/bash

MARGINAL=3
WORKLOAD=4096
WORKLOAD_SEED=0

MAX_IDXS=100
MAX_ITERS=100
SYNDATA_SIZE=1000

LR=1e-4

######

DATASET=acs_2010_1yr
STATE=PA

python gem_nondp.py --dataset $DATASET --state $STATE \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--lr $LR --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--syndata_size $SYNDATA_SIZE --verbose --overwrite

######

DATASET=acs_2018_1yr
STATE=CA

python gem_nondp.py --dataset $DATASET --state $STATE \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--lr $LR --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--syndata_size $SYNDATA_SIZE --verbose --overwrite

######

DATASET=acs_2018_1yr
STATE=OH

python gem_nondp.py --dataset $DATASET --state $STATE \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--lr $LR --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--syndata_size $SYNDATA_SIZE --verbose --overwrite