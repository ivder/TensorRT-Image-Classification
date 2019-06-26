# TensorRT-Image-Classification
C++ Visual Studio solution for Image Classification using Caffe Model and TensorRT inference platform.

# Tested on:
 - **TensorRT-5.1.5.0.Windows10.x86_64.cuda-10.0.cudnn7.5** (GA version)
 - Windows 10
 - Visual Studio 2017
 - OpenCV 3.4.0 with CUDA support
 - CUDA 10
 - cuDNN 7.5.0
 
# Usage
 - Install TensorRT for Windows
 - Make sure you have OpenCV with CUDA build support installed, and copy the world.dll file to \x64\Release\
 - Open solution, in **main.cpp** edit the path of your caffe model and input image
```
cv::Mat img = cv::imread("Image.jpg");
std::string model = "CaffeModel/deploy.prototxt";
std::string trained = "CaffeModel/network.caffemodel";
std::string mean = "CaffeModel/mean.binaryproto";
std::string label = "CaffeModel/labels.txt";
```
 - Build the solution on Release mode and run

# Result
```
Finding CUDA Device
Parsing Caffe Model
Building Cuda Engine
CUDA NO ERROR
Initialization Time : 28.576s
Classifying Image

TOP 1 Prediction
dog : 91.274193%


TOP 5 Predictions
cat : 0.912742%
bird : 0.0856047%
fish : 0.00106978%
person : 0.000507325%
reptile : 7.00301e-05%

Classification Time : 11ms
```
