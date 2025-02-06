#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

export PYTHONUNBUFFERED=1
python3 -m main.augment \
    --sentences-path ../test/multilingual/sentences.txt \
    --output-dir ../test/multilingual \
    --min-len 1 >../log/augment.log 2>&1
