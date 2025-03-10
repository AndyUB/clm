#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

num_gpus=8
data_pct=1
seq_len=200
batch_size=512
lr=0.0005
epochs=10

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/.cache/huggingface"
python3 -m main.dist \
    --mode train \
    --world-size $num_gpus \
    --data-dir ../data/tatoeba \
    --data-percentage $data_pct \
    --seq-len $seq_len \
    --batch-size $batch_size \
    --lr $lr \
    --epochs $epochs \
    --include-non-full-batches \
    --checkpoint-dir ../distwork/${num_gpus}gpus_${data_pct}tatoeba \
    --checkpoint-interval 1 \
    --report-interval 100 \
    --eval-result-dir ../distwork/${num_gpus}gpus_${data_pct}tatoeba \
    --k 3 >../log/train_gpt2_${num_gpus}gpus_${data_pct}tatoeba.log 2>&1
