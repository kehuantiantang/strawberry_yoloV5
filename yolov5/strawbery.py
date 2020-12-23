# coding=utf-8
import argparse
import torch

from pathlib import Path
import random
from torch.backends import cudnn
import time
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import strip_optimizer, set_logging, check_img_size, increment_path, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, load_classifier
import cv2.cv2
import numpy as np
from easydict import EasyDict as edict
import os.path as osp


class SurvedModel(object):

    def __init__(self):
        param = {
            #TODO please specify the weight path
            'weights':["./output/train/exp_907aug_894_deeper/weights/best.pt",
                       "./output/train/exp_913aug_885/weights/best.pt",
                       "./output/train/exp_904aug_886_l/weights/best.pt"],
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


    def detect(self, save_img=False) -> list:
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))


        # Initialize
        set_logging()
        device = select_device(self.opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

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
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        return_imgs = []
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

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
                        s += f'{n} {names[int(c)]}s, '  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
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
    array = '../dataset/make/test/angular_leafspot355.jpg'
    model = SurvedModel()

    return_img = model.predict(array)
    cv2.imwrite('test.jpg', return_img)


