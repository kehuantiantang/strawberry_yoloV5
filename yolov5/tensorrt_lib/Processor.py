import random

import cv2
import sys
import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import math
import time

import torch

from tensorrt_lib.post_detector import Detect
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box


class Processor():
    def __init__(self, model):
        # load tensorrt engine
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        TRTbin =  model
        print('trtbin', TRTbin)
        with open(TRTbin, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        # allocate memory
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append({ 'host': host_mem, 'device': device_mem })
            else:
                outputs.append({ 'host': host_mem, 'device': device_mem })
        # save to class
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream

        self.no = 12
        self.output_shapes = [
            (1, 3, 56, 56, self.no),
            (1, 3, 28, 28, self.no),
            (1, 3, 14, 14, self.no)
        ]
        self.names = ['angular_leafspot',
                      'anthracnose_fruit_rot',
                      'blossom_blight',
                      'gray_mold',
                      'leaf_spot',
                      'powdery_mildew_fruit',
                      'powdery_mildew_leaf']

        self.img_size = 448

    def get_preprocessing_image(self):
        return self.show_img

    def detect(self, img):
        resized = self.pre_process(img)
        outputs = self.inference(resized)
        reshaped = []
        for output, shape in zip(outputs, self.output_shapes):
            r = output.reshape(shape)
            reshaped.append(r)
        return reshaped

    def pre_process(self, img):
        print('original image shape', img.shape)
        img = cv2.resize(img, (self.img_size, self.img_size))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.transpose((2, 0, 1)).astype(np.float16)
        self.show_img = img.copy()
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img

    def inference(self, img):
        # copy img to input memory
        # self.inputs[0]['host'] = np.ascontiguousarray(img)
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        start = time.time()
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        end = time.time()
        print('Done execution time:', end-start)
        return [out['host'] for out in self.outputs]

    def pos_process(self, outputs):
        mydet = Detect(nc = len(self.names), anchors= [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]])
        output = mydet(outputs)

        pred = non_max_suppression(output, 0.1, 0.4, classes=None,
                                   agnostic=False)
        return pred


    def draw_bbox(self, pred, img, save_path):
        # Process detections
        t1 = time.time()
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        for i, det in enumerate(pred):  # detections per image

            s, im0 = '', img

            save_path = os.path.join(save_path, 'test_001.jpg')

            s += '%gx%g ' % img.shape[:-1]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[:-1], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {self.names[int(c)]}s, '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({time.time() - t1:.3f}s)')


            cv2.imwrite(save_path, im0)

    # def extract_object_grids(self, output):
    #     """
    #     Extract objectness grid
    #     (how likely a box is to contain the center of a bounding box)
    #     Returns:
    #         object_grids: list of tensors (1, 3, nx, ny, 1)
    #     """
    #     object_grids = []
    #     for out in output:
    #         probs = self.sigmoid_v(out[..., 4:5])
    #         object_grids.append(probs)
    #     return object_grids
    #
    # def extract_class_grids(self, output):
    #     """
    #     Extracts class probabilities
    #     (the most likely class of a given tile)
    #     Returns:
    #         class_grids: array len 3 of tensors ( 1, 3, nx, ny, 80)
    #     """
    #     class_grids = []
    #     for out in output:
    #         object_probs = self.sigmoid_v(out[..., 4:5])
    #         class_probs = self.sigmoid_v(out[..., 5:])
    #         obj_class_probs = class_probs * object_probs
    #         class_grids.append(obj_class_probs)
    #     return class_grids
    #
    # def extract_boxes(self, output, conf_thres=0.5):
    #     """
    #     Extracts boxes (xywh) -> (x1, y1, x2, y2)
    #     """
    #     scaled = []
    #     grids = []
    #     for out in output:
    #         out = self.sigmoid_v(out)
    #         _, _, width, height, _ = out.shape
    #         grid = self.make_grid(width, height)
    #         grids.append(grid)
    #         scaled.append(out)
    #     z = []
    #     for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
    #         _, _, width, height, _ = out.shape
    #         out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
    #         out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor
    #
    #         out[..., 5:] = out[..., 4:5] * out[..., 5:]
    #         out = out.reshape((1, 3 * width * height, self.no))
    #         z.append(out)
    #     pred = np.concatenate(z, 1)
    #     xc = pred[..., 4] > conf_thres
    #     pred = pred[xc]
    #     boxes = self.xywh2xyxy(pred[:, :4])
    #     return boxes

    # def post_process(self, outputs, conf_thres=0.5):
    #     """
    #     Transforms raw output into boxes, confs, classes
    #     Applies NMS thresholding on bounding boxes and confs
    #     Parameters:
    #         output: raw output tensor
    #     Returns:
    #         boxes: x1,y1,x2,y2 tensor (dets, 4)
    #         confs: class * obj prob tensor (dets, 1)
    #         classes: class type tensor (dets, 1)
    #     """
    #     scaled = []
    #     grids = []
    #     for out in outputs:
    #         out = self.sigmoid_v(out)
    #         _, _, width, height, _ = out.shape
    #         grid = self.make_grid(width, height)
    #         grids.append(grid)
    #         scaled.append(out)
    #     z = []
    #     for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
    #         _, _, width, height, _ = out.shape
    #         out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
    #         out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor
    #
    #         out = out.reshape((1, 3 * width * height, self.no))
    #         z.append(out)
    #     pred = np.concatenate(z, 1)
    #     xc = pred[..., 4] > conf_thres
    #     pred = pred[xc]
    #     return self.nms(pred)
    #
    # def make_grid(self, nx, ny):
    #     """
    #     Create scaling tensor based on box location
    #     Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
    #     Arguments
    #         nx: x-axis num boxes
    #         ny: y-axis num boxes
    #     Returns
    #         grid: tensor of shape (1, 1, nx, ny, 80)
    #     """
    #     nx_vec = np.arange(nx)
    #     ny_vec = np.arange(ny)
    #     yv, xv = np.meshgrid(ny_vec, nx_vec)
    #     grid = np.stack((yv, xv), axis=2)
    #     grid = grid.reshape(1, 1, ny, nx, 2)
    #     return grid
    #
    # def sigmoid(self, x):
    #     return 1 / (1 + math.exp(-x))
    #
    # def sigmoid_v(self, array):
    #     return np.reciprocal(np.exp(-array) + 1.0)
    # def exponential_v(self, array):
    #     return np.exp(array)

    # def non_max_suppression(self, boxes, confs, classes, iou_thres=0.6):
    #     x1 = boxes[:, 0]
    #     y1 = boxes[:, 1]
    #     x2 = boxes[:, 2]
    #     y2 = boxes[:, 3]
    #     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #     order = confs.flatten().argsort()[::-1]
    #     keep = []
    #     while order.size > 0:
    #         i = order[0]
    #         keep.append(i)
    #         xx1 = np.maximum(x1[i], x1[order[1:]])
    #         yy1 = np.maximum(y1[i], y1[order[1:]])
    #         xx2 = np.minimum(x2[i], x2[order[1:]])
    #         yy2 = np.minimum(y2[i], y2[order[1:]])
    #         w = np.maximum(0.0, xx2 - xx1 + 1)
    #         h = np.maximum(0.0, yy2 - yy1 + 1)
    #         inter = w * h
    #         ovr = inter / (areas[i] + areas[order[1:]] - inter)
    #         inds = np.where( ovr <= iou_thres)[0]
    #         order = order[inds + 1]
    #     boxes = boxes[keep]
    #     confs = confs[keep]
    #     classes = classes[keep]
    #     return boxes, confs, classes
    # 
    # def nms(self, pred, iou_thres=0.6):
    #     boxes = self.xywh2xyxy(pred[..., 0:4])
    #     # best class only
    #     confs = np.amax(pred[:, 5:], 1, keepdims=True)
    #     classes = np.argmax(pred[:, 5:], axis=-1)
    #     return self.non_max_suppression(boxes, confs, classes)
    # 
    # def xywh2xyxy(self, x):
    #     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #     y = np.zeros_like(x)
    #     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    #     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    #     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    #     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    #     return y