#!/bin/bash

if [[ ! "$PWD" =~ /clm/src$ ]]; then
    echo "Error: This script must be run from the 'clm/src' directory."
    exit 1
fi

export PYTHONUNBUFFERED=1
python3 -m main.gpt2_tatoeba \
    --mode predict \
    --model-path ../work/gpt2_0.05tatoeba \
    --input-path ../example/input.txt \
    --output-path ../pred/gpt2_tatoeba_pred.txt \
    --k 3 >../log/predict_gpt2_tatoeba.log 2>&1

# python3 -m main.gpt2_tatoeba \
#     --mode predict \
#     --model-path ../work/gpt2_0.05tatoeba \
#     --input-path ../test/multilingual/input.txt \
#     --output-path ../pred/multilingual/gpt2_tatoeba_pred.txt \
#     --k 3 >../log/predict_gpt2_tatoeba-multilingual.log 2>&1

# python3 -m main.gpt2_tatoeba \
#     --mode predict \
#     --model-path ../work/gpt2_0.05tatoeba \
#     --input-path ../test/adversarial/input.txt \
#     --output-path ../pred/adversarial/gpt2_tatoeba_pred.txt \
#     --k 3 >../log/predict_gpt2_tatoeba-adversarial.log 2>&1
