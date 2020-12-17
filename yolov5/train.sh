#!/usr/bin/env bash
python train.py --img-size 419 --sync-bn --multi-scale --workers 5 --batch-size 28 \
 --cfg "../strawberry_params/sb_model_larger.yaml" --epochs 500 --hyp "../strawberry_params/hyp.strawberry_large.yaml"