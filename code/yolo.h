#pragma once
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class YOLO {
public:
	struct DetectRet {
		float x1;
		float y1;
		float x2;
		float y2;
		float conf;
		std::string name;
	};

	YOLO(const std::string& engine_file, const std::string& class_file, const float& conf_threshold, const float& NMS_threshold);
	void Inference();

private:
	std::string ENGINE_FILE;
	std::string CLASS_FILE;
	float CONF_THRESHOLD;
	float NMS_THRESHOLD;
	int BATCH_SIZE;
	int IMAGE_HEIGHT;
	int IMAGE_WIDTH;
	int IMAGE_CHANNEL;
	int MAX_IMAGE_SIZE;
	int NUM_BOXES;
	int NUM_CLASSES;
	int NUM_OBJECTS;
	std::map<int, std::string> LABELS;

	nvinfer1::ICudaEngine* engine = nullptr;
	nvinfer1::IExecutionContext* context = nullptr;
};
