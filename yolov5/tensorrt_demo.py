import time

import cv2
import argparse

import os

from tensorrt_lib.Processor import Processor
from tensorrt_lib.Visualizer import Visualizer


def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model',
                        default='/Strawberry/yolov5/output/train/exp_893aug_887_m/weights/best_ap05.trt', help='trt engine file located in ./models', required=False)
    parser.add_argument('-i', '--image', default='/Strawberry/dataset/make/test/blossom_blight199.jpg', help='image file path', required=False)

    parser.add_argument('--save_path', default='./output/detect/exp')
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = cli()

    os.makedirs(args.save_path, exist_ok=True)
    # setup processor and visualizer

    processor = Processor(model=args.model)
    # visualizer = Visualizer(args.save_path)

    # fetch input
    print('image arg', args.image)
    img = cv2.imread(args.image)

    # inference

    # for i in range(1000):
    outputs = processor.detect(img)
    print([p.shape for p in outputs])

    pre_img = processor.get_preprocessing_image()

    pred = processor.pos_process(outputs)
    print([p.size() for p in pred])
    processor.draw_bbox(pred, pre_img, args.save_path)



    # ref https://github.com/SeanAvery/yolov5-tensorrt
    # https://github.com/bei91/yolov5-onnx-tensorrt/blob/master/demo/onnx_tensorrt.py


if __name__ == '__main__':
    main()   
