#pragma once

struct AffineMatrix {
	float value[6];
};

void preProcess(uint8_t* src, const int& src_height, const int& src_width, const float& scale,
	float* dst, const int& dst_height, const int& dst_width);

void postProcess(float* host_input, const int& buffer_size, const int& num_boxes,
	const int& num_classes, const float& conf_threshold, const float& NMS_threshold, const float& ratio, 
	const int& num_objects, std::map<int, std::string>& labels, std::vector<YOLO::DetectRet>& detections);
