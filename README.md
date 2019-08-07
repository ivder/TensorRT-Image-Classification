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
 - Check Project Properties and make sure all dependencies paths (TensorRT, CUDA, OpenCV) are correct
 - Build the solution on Release mode and run
 - The first time you running this program, it will take some time to build the CUDA engine, and the engine will be saved as **ClassificationTRT.engine**. The next time you run the program, it will load the created engine in a short time.

# Result
```
Finding CUDA Device
Parsing Caffe Model
Loading ClassificationTRT.engine
CUDA NO ERROR
Initialization Time : 3.646s
Classifying Image

TOP 1 Prediction
pothole : 91.274185%


TOP 5 Predictions
pothole : 91.274185%
shadow : 8.560489%
patch : 0.106978%
patchdamaged : 0.050732%
paintasphalt : 0.007003%

Classification Time : 10ms
```
