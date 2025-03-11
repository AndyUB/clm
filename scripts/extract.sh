#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

checkpoint_dir="../distwork/8gpus_1tatoeba_epochs11-19"
checkpoint_name=epoch17
checkpoint_file="${checkpoint_name}.pt"
output_dir="../work/best/${checkpoint_name}"

mkdir -p $output_dir

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/.cache/huggingface"
python3 -m main.extract \
    --checkpoint-dir $checkpoint_dir \
    --checkpoint-file $checkpoint_file \
    --output-dir $output_dir \
    >../log/extract.log 2>&1
