#!/usr/bin/env bash
python test.py --data "../strawberry_params/strawberry.yaml" --img-size 419 \
--iou-thres 0.5 --weight "./output/train/exp_907aug_894_deeper/weights/best.pt" "./output/train/exp_913aug_885/weights/best.pt" "./output/train/exp_904aug_886_l/weights/best.pt" --verbose
