# YOLOv8
The Pytorch implementation is ultralytics/ultralytics

# Requirements
CUDA 11.6 \n
CUDNN 8.8.x
TensorRT 8.4
OpenCV 4.6.0

# How to Run, yolov8s as example
1. generate .onnx from pytorch with .pt
2. Open ONNX2TensorRT project and run it
3. Open yolov8 project and run it

# INT8 Quantization
set the macro USE_INT8 in config.h and make
serialize the model and test


# 转载请注明出处
