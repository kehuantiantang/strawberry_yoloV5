# coding=utf-8
import tensorrt as trt
import sys
import argparse
import os
"""
takes in onnx model
converts to tensorrt
"""

def cli():
    desc = 'compile Onnx model to TensorRT'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', help='onnx file location inside ./lib/models', default =
    '/Strawberry/yolov5/output/train/exp_893aug_887_m/weights/best_ap05_sim.onnx')
    parser.add_argument('-fp', '--floatingpoint', type=int, default=32, help='floating point precision. 16 or 32')
    parser.add_argument('-o', '--output', help='name of trt output file', default = 'output/detect/exp')
    args = parser.parse_args()
    model = args.model or 'yolov5s-simple.onnx'
    fp = args.floatingpoint
    if fp != 16 and fp != 32:
        print('floating point precision must be 16 or 32')
        sys.exit()
    output = args.output or 'yolov5s-simple-{}.trt'.format(fp)
    return {
        'model': model,
        'fp': fp,
        'output': output
    }

if __name__ == '__main__':
    args = cli()
    batch_size = 1
    model = args['model']
    output = args['output']
    logger = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # trt7
    with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = batch_size
        if args['fp'] == 16:
            builder.fp16_mode = True
        with open(model, 'rb') as f:
            print('Beginning ONNX file parsing')
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("ERROR", parser.get_error(error))
        print("num layers:", network.num_layers)

        # network.get_input(0).shape = [batch_size, 3, 448, 448]  # trt7
        network.get_input(0).shape = [batch_size, 3, 448, 448]  # trt7
        last_layer = network.get_layer(network.num_layers - 1)

        # print('last layer', last_layer.get_output(0))
        # if not last_layer.get_output(0):
        # network.mark_output(last_layer.get_output(0))

        # reshape input from 32 to 1
        engine = builder.build_cuda_engine(network)
        with open(model.replace('onnx', 'trt'), 'wb') as f:
            f.write(engine.serialize())
        #
        print('Successfuly save to ', model.replace('onnx', 'trt'))
        print("Completed creating Engine")