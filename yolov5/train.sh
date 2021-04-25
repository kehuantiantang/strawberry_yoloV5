#!/usr/bin/env bash
#python train.py --img-size 419 --workers 16 --batch-size 20 --cfg "../strawberry_params/sb_model_x.yaml" \
#--epochs 1000 --weight "/home/ailab/dataset/yolo/Strawberry/yolov5/output/train/exp_913aug_885/weights/best_ap05.pt" \
#--hyp "../strawberry_params/hyp.strawberry.yaml" --multi-scale

#python train.py --img-size 448 --workers 16 --batch-size 32 \
# --cfg "../strawberry_params/sb_model_l.yaml" --epochs 1000 --hyp "../strawberry_params/hyp.strawberry.yaml" \
#--weight "output/train/exp_911aug_889_l/weights/best.pt"
# --multi-scale


python train.py --img-size 448 --workers 16 --batch-size 6 \
 --cfg "../strawberry_params/sb_model_x.yaml" --epochs 1000 --hyp "../strawberry_params/hyp.strawberry.yaml" \
 --multi-scale