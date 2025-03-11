#!/usr/bin/env bash
set -e
set -v
python src/best_model.py test --work_dir work/best --test_data $1 --test_output $2
