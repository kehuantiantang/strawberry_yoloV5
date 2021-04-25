import time

import cv2
import argparse
from tensorrt_lib.Processor import Processor
from tensorrt_lib.Visualizer import Visualizer


def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='/Strawberry/yolov5/output/train/exp_893aug_887_m/weights/best_ap05.trt', help='trt engine file located in ./models', required=False)
    parser.add_argument('-i', '--image', default='/Strawberry/dataset/make/test/blossom_blight199.jpg', help='image file path', required=False)
    args = parser.parse_args()
    return { 'model': args.model, 'image': args.image }

def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer

    processor = Processor(model=args['model'])
    visualizer = Visualizer()

    # fetch input
    print('image arg', args['image'])
    img = cv2.imread(args['image'])

    # inference

    for i in range(1000):
        output = processor.detect(img)

    # TODO NMS and draw bbox to image
    # ref https://github.com/SeanAvery/yolov5-tensorrt
    # https://github.com/bei91/yolov5-onnx-tensorrt/blob/master/demo/onnx_tensorrt.py

    import sys
    sys.exit(0)
    # img = cv2.resize(img, (640, 640))
    #
    # # object visualization
    # object_grids = processor.extract_object_grids(output)
    # visualizer.draw_object_grid(img, object_grids, 0.1)
    #
    # # class visualization
    # class_grids = processor.extract_class_grids(output)
    # visualizer.draw_class_grid(img, class_grids, 0.01)
    #
    # # bounding box visualization
    # boxes = processor.extract_boxes(output)
    # visualizer.draw_boxes(img, boxes)
    #
    # # final results
    # boxes, confs, classes = processor.post_process(output)
    # visualizer.draw_results(img, boxes, confs, classes)

if __name__ == '__main__':
    main()   
