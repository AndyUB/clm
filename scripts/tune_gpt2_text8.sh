#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

export PYTHONUNBUFFERED=1
python3 -m main.gpt2_text8 \
    --mode hyperparam \
    --data-path ../data/text8 \
    --data-percentage 0.0001 \
    --output-path ../work \
    --k 3 >../log/tune_gpt2_text8.log 2>&1
