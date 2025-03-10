#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

data_pct=0.1
world_size=8

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/.cache/huggingface"
python3 -m main.dist \
    --mode tune \
    --world-size $world_size \
    --data-dir ../data/tatoeba \
    --data-percentage $data_pct \
    --checkpoint-dir ../disttune/${data_pct}tatoeba_${world_size}gpu \
    --eval-result-dir ../disttune/${data_pct}tatoeba_${world_size}gpu \
    --k 3 >../log/tune_gpt2_${data_pct}tatoeba_${world_size}gpu.log 2>&1
