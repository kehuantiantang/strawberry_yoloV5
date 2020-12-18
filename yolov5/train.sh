#!/usr/bin/env bash
python train.py --img-size 419 --sync-bn --workers 8 --batch-size 64 \
 --cfg "../strawberry_params/sb_model_l.yaml" --epochs 500 --hyp "../strawberry_params/hyp.strawberry.yaml" \
--weight "../dataset/yolov5l.pt" --multi-scale