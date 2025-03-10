#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

export PYTHONUNBUFFERED=1
python3 -m main.gpt2_enwik8 \
    --mode predict \
    --model-path ../work/gpt2_0.1enwik8_dirty \
    --input-path ../example/input.txt \
    --output-path ../pred/gpt2_enwik8_pred.txt \
    --k 3 >../log/predict_gpt2_enwik8.log 2>&1
