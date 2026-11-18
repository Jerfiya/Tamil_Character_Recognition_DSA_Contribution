#!/usr/bin/env bash
# Batch 2: remaining final runs (ArcFace -> episodic fine-tune, ProtoNet extra seeds), then the
# LR comparison and the validation-only ablations (ProtoNet, Protocol B, 8 epochs each).
# Usage: bash run_batch2.sh   (5 runs in parallel, 2 threads each)
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
ARC=$(ls -d runs/*_arcface_conv4_B_final | tail -1)/checkpoints/best_val_loss.pt
[ -f "$ARC" ] || { echo "ArcFace checkpoint missing: $ARC"; exit 1; }
F="train.epochs=15 val.episodes=100 method.name=protonet train.protocol=B"
A="train.epochs=8 val.episodes=100 method.name=protonet train.protocol=B"
jobs=(
 "arcface_ft_B|src/train_fewshot.py --tag arcft_final --set $F method.init_from=$ARC train.lr=0.0003 train.warmup_epochs=1"
 "protonet_B_seed7|src/train_fewshot.py --tag final_seed7 --set $F seed=7"
 "protonet_B_seed123|src/train_fewshot.py --tag final_seed123 --set $F seed=123"
 "lr_3e-3|src/train_fewshot.py --tag lr3e-3 --set $A train.lr=0.003"
 "lr_1e-3|src/train_fewshot.py --tag lr1e-3 --set $A train.lr=0.001"
 "lr_3e-4|src/train_fewshot.py --tag lr3e-4 --set $A train.lr=0.0003"
 "lr_1e-4|src/train_fewshot.py --tag lr1e-4 --set $A train.lr=0.0001"
 "abl_way5|src/train_fewshot.py --tag abl_way5 --set $A train.n_way=5"
 "abl_way10|src/train_fewshot.py --tag abl_way10 --set $A train.n_way=10"
 "abl_way30|src/train_fewshot.py --tag abl_way30 --set $A train.n_way=30"
 "abl_shot1|src/train_fewshot.py --tag abl_shot1 --set $A train.k_shot=1"
 "abl_shotmix|src/train_fewshot.py --tag abl_shotmix --set $A train.k_shot=[1,2,3,5]"
 "abl_emb64|src/train_fewshot.py --tag abl_emb64 --set $A model.emb_dim=64"
 "abl_emb256|src/train_fewshot.py --tag abl_emb256 --set $A model.emb_dim=256"
 "abl_cosine|src/train_fewshot.py --tag abl_cosine --set $A method.distance=cosine"
 "abl_noema|src/train_fewshot.py --tag abl_noema --set $A train.ema=false"
 "abl_in64|src/train_fewshot.py --tag abl_in64 --set $A model.input_size=64"
 "abl_resnet12|src/train_fewshot.py --tag abl_resnet12 --set $A model.backbone=resnet12"
 "abl_mnv3_pre96|src/train_fewshot.py --tag abl_mnv3_pre96 --set $A model.backbone=mobilenetv3_large model.pretrained=true model.input_size=96"
 "abl_mnv3_scratch96|src/train_fewshot.py --tag abl_mnv3_scr96 --set $A model.backbone=mobilenetv3_large model.pretrained=false model.input_size=96"
)
printf '%s\n' "${jobs[@]}" | xargs -P 5 -I{} bash -c 'name="${1%%|*}"; cmd="${1#*|}"; '"$PY"' $cmd > logs/$name.log 2>&1; echo "$name exit=$?"' _ {}
