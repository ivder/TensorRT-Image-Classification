#include <iostream>
#include <chrono>
#include "classification.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() 
{
	static ClassificationTensorRT CLASSIFICATION_TENSORRT;
	// Edit image and caffe model path below
	cv::Mat img = cv::imread("image.jpg");
	std::string model = "CaffeModel/deploy.prototxt";
	std::string trained = "CaffeModel/network.caffemodel";
	std::string mean = "CaffeModel/mean.binaryproto";
	std::string label = "CaffeModel/labels.txt";
	ClassificationTensorRT::classifier_ctx *ctx;
	int time1, time2;
	
	time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	ctx = CLASSIFICATION_TENSORRT.classifier_initialize(model, trained, mean, label);
	time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;

	time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	const char* cstr = CLASSIFICATION_TENSORRT.classifier_classify(ctx, img);
	time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	std::cout << "Classification Time : " << time2 - time1 << "ms" << std::endl;

	system("PAUSE");
	return 0;
}
