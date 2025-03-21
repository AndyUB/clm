#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

export PYTHONUNBUFFERED=1
python3 -m main.gpt2_enwik8 \
    --mode train \
    --data-path ../data/enwik8 \
    --data-percentage 0.1 \
    --output-path ../work \
    --k 3 >../log/train_gpt2_enwik8.log 2>&1
