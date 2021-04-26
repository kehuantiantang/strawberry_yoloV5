This repository strongly refers to the [yolov5](https://github.com/ultralytics/yolov5), and add some features for 
onnx, onnx-sim, tensorrt conversion.

### Development pipeline
- [x] Convert `*.pt` to onnx, refer to [this](./onnx_lib/export.py)
- [x] Simplify the `*.onnx` to `*_sim.onnx`, please install the library onnx_simplifier first.
```s
 python -m onnxsim /Strawberry/output/train/exp_893aug_887_m/weights/best_ap05_32.onnx /Strawberry/output/train/exp_893aug_887_m/weights/best_ap05_sim_32.onnx
```
- [x] Test the inference result in `*.onnx` and `*_sim.onnx`, in [here](./strawbery_onnx_detect.py)
- [x] Convert `*.onnx` to `*.trt`, TensorRT is widely used in accelerate the inference time. the script shows in 
  [here](./tensorrt_lib/export_tensorrt.py)
- [x] Test the Inference result in `*.trt*`, [here](tensorrt_demo.py)
- [ ] Accuracy decay in `*.trt`,  the problem caused by int64 to int32 conversion, analyse the  onnx attribute type 
  may helpful to address this problem.
  
---
### Time comparison
|Model|Inference |Initialization|Total(Init+Infer+NMS+DrawBBox)|
|---|---|---|---|
|origin|0.121s|0.3625s|1.965s|
|*.onnx|0.102s|0.2329s|1.952s|
|*_sim.onnx|0.097s|0.2091s|1.947s|
|*.trt (sim)|0.0073s|12.095s|12.127s|


### The suggestion of reducing time
- Initialize the model while web server start
- NMS, pytorch version --> numpy
- Large batchsize --> tensorRT





