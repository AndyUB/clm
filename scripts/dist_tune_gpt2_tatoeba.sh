#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

data_pct=0.001

export PYTHONUNBUFFERED=1
python3 -m main.dist \
    --mode tune \
    --world-size 2 \
    --data-dir ../data/tatoeba \
    --data-percentage $data_pct \
    --checkpoint-dir ../disttune/${data_pct}tatoeba \
    --eval-result-dir ../disttune/${data_pct}tatoeba \
    --k 3 >../log/tune_gpt2_${data_pct}tatoeba.log 2>&1
