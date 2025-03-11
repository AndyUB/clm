#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

output_dir="../test/gen-ai/augmented"
mkdir -p $output_dir

export PYTHONUNBUFFERED=1
python3 -m main.augment \
    --input-path "../test/gen-ai/input.txt" \
    --answer-path "../test/gen-ai/answer.txt" \
    --output-dir $output_dir \
    --min-len 1 >"../log/augment_gen-ai.log" 2>&1
