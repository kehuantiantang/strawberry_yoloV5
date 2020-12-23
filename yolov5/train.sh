#!/usr/bin/env bash
python train.py --img-size 419 --sync-bn --workers 8 --batch-size 32 \
 --cfg "../strawberry_params/sb_model_m.yaml" --epochs 500 --hyp "../strawberry_params/hyp.strawberry.yaml" \
--weight "output/train/exp/weights/best.pt" --multi-scale