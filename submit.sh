#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Eliana Dietrich,eliana1\nJay Bhateja,jbhateja\nAndy Ruan,yhruan22" >submit/team.txt

# train model
# python src/best_model.py train --work_dir work

# make predictions on example data submit it in pred.txt
model_dir=work/best
python src/best_model.py test --work_dir $model_dir --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
submit_model_dir=submit/$model_dir
mkdir -p submit/work
cp -r $model_dir $submit_model_dir

# remove pycache
find submit -type d -name "__pycache__" -exec rm -r {} +

# make zip file
zip -r submit.zip submit
