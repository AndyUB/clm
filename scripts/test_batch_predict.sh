#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

input_dir=../test/test_batch_prediction
mkdir -p $input_dir
output_dir=../pred/test_batch_prediction
mkdir -p $output_dir

lengths=(
    "1 200"
    "50 100"
    "200 200"
)

export PYTHONUNBUFFERED=1
for length in "${lengths[@]}"; do
    set -- $length
    min_len=$1
    max_len=$2

    python -m test.test_batch_prediction \
        --model-path ../work/gpt2_0.05tatoeba \
        --input-dir $input_dir \
        --output-path $output_dir/$min_len-$max_len.txt \
        --min-len "$min_len" \
        --max-len "$max_len" \
        --k 3 >../log/test_batch_prediction_$min_len-$max_len.log 2>&1
done
