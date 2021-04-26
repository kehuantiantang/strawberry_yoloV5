# coding=utf-8
import argparse
import torch

from pathlib import Path
import random
from torch.backends import cudnn
import time
from models.experimental import attempt_load
from tensorrt_lib.post_detector import Detect
from utils.datasets import LoadImages, LoadStreams
from utils.general import strip_optimizer, set_logging, check_img_size, increment_path, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, load_classifier
import cv2.cv2
import numpy as np
from easydict import EasyDict as edict
import os.path as osp
import onnx
import onnxruntime


class SurvedModel(object):

    def __init__(self):
        param = {
            #TODO please specify the weight path
            'weights':["./output/train/exp_893aug_887_m/weights/best_ap05.onnx"],
            'source':None,
            'img_size': 448,
            'conf_thres': 0.4,
            'iou_thres': 0.5,
            # TODO you can specify the gpu device here
            'device': '0',
            'view_img': False,
            'save_txt': False,
            'save_conf': False,
            'classes': None,
            'agnostic_nms': False,
            'augment': True,
            'update': False,
            'project': 'output/detect',
            'name': 'exp',
            'exist_ok': True,
            'save_dir':None
        }
        self.names = ['angular_leafspot',
                      'anthracnose_fruit_rot',
                      'blossom_blight',
                      'gray_mold',
                      'leaf_spot',
                      'powdery_mildew_fruit',
                      'powdery_mildew_leaf']

        self.opt = edict(param)

    def predict (self, img:np.array) -> np.array:

        # increment run
        self.opt.save_dir = Path(increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))

        # Directories
        (self.opt.save_dir / 'labels' if self.opt.save_txt else self.opt.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # temp_path = osp.join(self.opt.save_dir, 'predict.jpg')
        # cv2.imwrite(temp_path, img)

        self.opt.source = img
        # self.opt.source = temp_path
        with torch.no_grad():
            return_imgs = self.detect()
            return return_imgs[0]


    def det_inference(self, x):
        mydet = Detect(nc = len(self.names), anchors= [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]])
        output = mydet(x)
        # print(output)
        return output

    def detect(self, save_img=False) -> list:
        t_0 = time.time()
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))


        # Initialize
        set_logging()
        # device = select_device(self.opt.device)
        # half = device.type != 'cpu'  # half precision only supported on CUDA
        device = 'cpu'

        # Load model

        ort_session = onnxruntime.InferenceSession(weights[0])


        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors

        colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        print(time.time() - t_0)
        # Run inference
        t0 = time.time()

        return_imgs = []
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            # img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img.float()
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if img.shape[0] == 3:
                img = img.unsqueeze(0)
            img = img.numpy()

            # Inference
            t1 = time_synchronized()
            # pred = model(img, augment=self.opt.augment)[0]
            outname = [output.name for output in ort_session.get_outputs()]
            inname = [input.name for input in ort_session.get_inputs()]

            pred = ort_session.run(outname, {inname[0]: img})
            # pred = torch.cat([torch.from_numpy(i.reshape(1, -1, 12) )for i in pred], dim = 1)
            pred = self.det_inference(pred)

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t2 = time_synchronized()


            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = Path(path[i]), '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)

                save_path = str(self.opt.save_dir / p.name)
                txt_path = str(self.opt.save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {self.names[int(c)]}s, '  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        return_imgs.append(im0.copy())

        if save_txt or save_img:
            s = f"\n{len(list(self.opt.save_dir.glob('labels/*.txt')))} labels saved to {self.opt.save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {self.opt.save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        return return_imgs



if __name__ == '__main__':
    array = '../dataset/make/test/blossom_blight199.jpg'

    # weight_path = ''
    # # model check
    # # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    # onnx_model = onnx.load(weight_path)
    # onnx.checker.check_model(onnx_model)
    #
    # ort_session = onnxruntime.InferenceSession(weight_path)
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # ort_outs = ort_session.run(None, ort_inputs)

    model = SurvedModel()
    return_img = model.predict(array)
    cv2.imwrite('test.jpg', return_img)


