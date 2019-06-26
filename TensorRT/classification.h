#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H
#include <stddef.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef __cplusplus
extern "C" {
#endif


class ClassificationTensorRT {
public:
	typedef struct classifier_ctx classifier_ctx;

	classifier_ctx* classifier_initialize(std::string model_file, std::string trained_file, std::string mean_file, std::string label_file);

	const char* classifier_classify(classifier_ctx* ctx, cv::Mat img);

	void classifier_destroy(classifier_ctx* ctx);

};

#ifdef __cplusplus
}
#endif

#endif