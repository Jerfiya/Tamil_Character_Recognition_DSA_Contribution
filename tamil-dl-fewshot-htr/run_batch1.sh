#!/usr/bin/env bash
# Batch 1 (session 2): resume interrupted runs, re-run fixed Siamese/MAML, epoch-budget study.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
jobs=(
 "arcface_B_resume|src/train_fewshot.py --resume runs/20261116_185222_arcface_conv4_B_final"
 "baselinepp_B_resume|src/train_fewshot.py --resume runs/20261116_185539_baselinepp_conv4_B_final"
 "siamese_B|src/train_fewshot.py --tag final --set train.epochs=15 val.episodes=100 method.name=siamese train.protocol=B"
 "maml_B|src/train_fewshot.py --tag final --set train.epochs=8 val.episodes=100 method.name=maml train.protocol=B"
 "epochstudy_A|src/train_fewshot.py --tag epochstudy --set train.epochs=40 train.early_stopping=false val.episodes=100 method.name=protonet train.protocol=A"
)
printf '%s\n' "${jobs[@]}" | xargs -P 5 -I{} bash -c 'name="${1%%|*}"; cmd="${1#*|}"; '"$PY"' $cmd > logs/$name.log 2>&1; echo "$name exit=$?"' _ {}
