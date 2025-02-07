#!/usr/bin/env bash
set -e
set -v
python src/best_model.py test --work_dir work/gpt2_0.05tatoeba --test_data $1 --test_output $2
