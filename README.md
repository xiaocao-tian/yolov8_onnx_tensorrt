# YOLOv8
The Pytorch implementation is [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)


# Requirements
CUDA 11.8 <br />
CUDNN 8.8.x <br />
TensorRT 8.6 <br />
OpenCV 4.8.0 <br />

# How to Run, yolov8s as example
1. generate .onnx from pytorch with .pt
2. Open ONNX2TensorRT project and run it
3. Open yolov8 project and run it

# INT8 Quantization
set the macro USE_INT8 in config.h and make
serialize the model and test


# 转载请注明出处
