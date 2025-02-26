#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

export PYTHONUNBUFFERED=1
python3 -m main.gpt2_tatoeba \
    --mode hyperparam \
    --data-path ../data/tatoeba \
    --data-percentage 0.00001 \
    --output-path ../work \
    --k 3 >../log/tune_gpt2_tatoeba.log 2>&1
