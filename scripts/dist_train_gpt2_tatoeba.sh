#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

data_pct=0.00001

export PYTHONUNBUFFERED=1
python3 -m main.dist \
    --mode train \
    --world-size 2 \
    --data-dir ../data/tatoeba \
    --data-percentage $data_pct \
    --include-non-full-batches \
    --checkpoint-dir ../distwork/${data_pct}tatoeba \
    --report-interval 1 \
    --eval-result-dir ../distwork/${data_pct}tatoeba \
    --k 3 >../log/train_gpt2_${data_pct}tatoeba.log 2>&1
