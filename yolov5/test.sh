#!/usr/bin/env bash
python test.py --data "../strawberry_params/strawberry.yaml" --img-size 419 \
--iou-thres 0.5 --weight "./output/train/exp/weights/best.pt"  --verbose