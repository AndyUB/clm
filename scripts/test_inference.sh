#!/bin/bash

if [[ ! "$PWD" =~ /clm$ ]]; then
    echo "Error: This script must be run from the 'clm' directory."
    exit 1
fi

input_dirs=(
    "example"
    "test/adversarial"
    "test/gen-ai"
    "test/gen-ai/augmented"
    "test/multilingual"
)

epoch=$1
work_dir="work/test/epoch${epoch}"
pred_dir="pred/epoch${epoch}"
log_dir="log"
log_file="${log_dir}/inference_o_epoch${epoch}.log"

mkdir -p $pred_dir
mkdir -p $log_dir
>$log_file

# Remove if not needed
export HF_HOME="$HOME/.cache/huggingface"

export PYTHONUNBUFFERED=1
for input_dir in "${input_dirs[@]}"; do
    input_file="${input_dir}/input.txt"
    output_prefix="${input_dir//\//-}" # Replace all "/" with "-"
    output_file="${pred_dir}/${output_prefix}.txt"

    python3 src/best_model.py test \
        --work_dir $work_dir \
        --test_data $input_file \
        --test_output $output_file >>$log_file 2>&1

    answer_file="${input_dir}/answer.txt"
    if [ -e $answer_file ]; then
        python3 grader/grade.py $output_file $answer_file >>$log_file 2>&1
    fi
done
