#!/usr/bin/env bash
# Batch 3 (session 3): finish everything interrupted when session 2 was killed.
# Resumes the partial runs exactly (checkpoints/last.pt), restarts the runs that never logged an epoch,
# then the LR comparison and validation-only ablations. Longest jobs first. 6 runs in parallel x 2 threads.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
ARC=$(ls -d runs/*_arcface_conv4_B_final | tail -1)/checkpoints/best_val_loss.pt
[ -f "$ARC" ] || { echo "ArcFace checkpoint missing: $ARC"; exit 1; }
F="train.epochs=15 val.episodes=100 method.name=protonet train.protocol=B"
A="train.epochs=8 val.episodes=100 method.name=protonet train.protocol=B"
jobs=(
 "epochstudy_A_resume|src/train_fewshot.py --resume runs/20261116_222957_protonet_conv4_A_epochstudy"
 "arcface_ft_B|src/train_fewshot.py --tag arcft_final --set $F method.init_from=$ARC train.lr=0.0003 train.warmup_epochs=1"
 "protonet_B_seed7|src/train_fewshot.py --tag final_seed7 --set $F seed=7"
 "protonet_B_seed123|src/train_fewshot.py --tag final_seed123 --set $F seed=123"
 "siamese_B_resume|src/train_fewshot.py --resume runs/20261116_222957_siamese_conv4_B_final"
 "baselinepp_B_resume2|src/train_fewshot.py --resume runs/20261116_185539_baselinepp_conv4_B_final"
 "maml_B|src/train_fewshot.py --tag final --set train.epochs=8 val.episodes=100 method.name=maml train.protocol=B"
 "abl_mnv3_pre96|src/train_fewshot.py --tag abl_mnv3_pre96 --set $A model.backbone=mobilenetv3_large model.pretrained=true model.input_size=96"
 "abl_mnv3_scr96|src/train_fewshot.py --tag abl_mnv3_scr96 --set $A model.backbone=mobilenetv3_large model.pretrained=false model.input_size=96"
 "abl_effb0_pre96|src/train_fewshot.py --tag abl_effb0_pre96 --set $A model.backbone=efficientnet_b0 model.pretrained=true model.input_size=96"
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
 "abl_resnet18|src/train_fewshot.py --tag abl_resnet18 --set $A model.backbone=resnet18"
)
printf '%s\n' "${jobs[@]}" | xargs -P 6 -I{} bash -c 'name="${1%%|*}"; cmd="${1#*|}"; '"$PY"' $cmd > logs/$name.log 2>&1; echo "$(date +%H:%M) $name exit=$?"' _ {}
echo "BATCH3 DONE $(date +%H:%M)"
