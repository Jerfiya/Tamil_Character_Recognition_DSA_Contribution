#!/usr/bin/env bash
# Trains every method (final runs). Usage: bash run_all.sh   (5 runs in parallel, 2 threads each)
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
jobs=(
 "protonet_B|src/train_fewshot.py --tag final --set train.epochs=15 val.episodes=100 method.name=protonet train.protocol=B"
 "protonet_A|src/train_fewshot.py --tag final --set train.epochs=15 val.episodes=100 method.name=protonet train.protocol=A"
 "supervised_A|src/train_baseline.py --tag final --set train.epochs=15 val.episodes=100"
 "matchingnet_B|src/train_fewshot.py --tag final --set train.epochs=15 val.episodes=100 method.name=matchingnet train.protocol=B"
 "relationnet_B|src/train_fewshot.py --tag final --set train.epochs=15 val.episodes=100 method.name=relationnet train.protocol=B"
 "siamese_B|src/train_fewshot.py --tag final --set train.epochs=15 val.episodes=100 method.name=siamese train.protocol=B"
 "arcface_B|src/train_fewshot.py --tag final --set train.epochs=15 val.episodes=100 method.name=arcface train.protocol=B"
 "baselinepp_B|src/train_fewshot.py --tag final --set train.epochs=15 val.episodes=100 method.name=baselinepp train.protocol=B"
 "maml_B|src/train_fewshot.py --tag final --set train.epochs=8 val.episodes=100 method.name=maml train.protocol=B"
)
printf '%s\n' "${jobs[@]}" | xargs -P 5 -I{} bash -c 'name="${1%%|*}"; cmd="${1#*|}"; '"$PY"' $cmd > logs/$name.log 2>&1; echo "$name exit=$?"' _ {}
# --- evaluate every finished run once on the test set (4 in parallel), then build the report
ls -d runs/*_final | xargs -P 4 -I{} bash -c '"$0" src/evaluate.py "$1" > "logs/eval_$(basename "$1").log" 2>&1; echo "eval $(basename "$1") exit=$?"' "$PY" {}
$PY src/summarize.py > logs/summarize.log 2>&1; echo "summarize exit=$?"
