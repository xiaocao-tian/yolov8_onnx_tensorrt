#include "yolo.h"
#include "process.h"

__global__ void warpAffine (
	uint8_t* src, int src_line_size, int src_height, int src_width,
	float* dst, int dst_height, int dst_width, int constant, AffineMatrix d2s, int jobs) {
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= jobs) return;

	float m_x1 = d2s.value[0];
	float m_y1 = d2s.value[1];
	float m_x2 = d2s.value[3];
	float m_y2 = d2s.value[4];

	int dx = position % dst_width;
	int dy = position / dst_height;
	float src_x = m_x1 * dx + m_y1 * dy;
	float src_y = m_x2 * dx + m_y2 * dy;
	float c0, c1, c2;

	if (src_x < 0 || src_x + 1 >= src_width || src_y < 0 || src_y + 1 >= src_height) {
		c0 = constant;
		c1 = constant;
		c2 = constant;
	}
	else {
		int x_low = floorf(src_x);
		int y_low = floorf(src_y);
		int x_high = x_low + 1;
		int y_high = y_low + 1;
		float w1 = (y_high - src_y) * (x_high - src_x);
		float w2 = (y_high - src_y) * (src_x - x_low);
		float w3 = (src_y - y_low) * (x_high - src_x);
		float w4 = (src_y - y_low) * (src_x - x_low);
		uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
		uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
		uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
		uint8_t* v4 = src + y_high * src_line_size + x_high * 3;
		c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
		c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
		c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
	}

	// bgr -> rgb
	float temp = c2;
	c2 = c0;
	c0 = temp;

	// normalization
	c0 /= 255.0f;
	c1 /= 255.0f;
	c2 /= 255.0f;

	// rgbrgbrgb -> rrrgggbbb
	int area = dst_height * dst_width;
	float* pdst_c0 = dst + dy * dst_width + dx;
	float* pdst_c1 = pdst_c0 + area;
	float* pdst_c2 = pdst_c1 + area;
	*pdst_c0 = c0;
	*pdst_c1 = c1;
	*pdst_c2 = c2;
}


void preProcess(uint8_t* src, const int& src_height, const int& src_width, const float& scale,
				float* dst, const int& dst_height, const int& dst_width) {
	AffineMatrix s2d, d2s;

	s2d.value[0] = scale;
	s2d.value[1] = 0;
	s2d.value[2] = 0;
	s2d.value[3] = 0;
	s2d.value[4] = scale;
	s2d.value[5] = 0;

	cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
	cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
	cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

	
	int jobs = dst_height * dst_width;
	int threads = 512;
	int blocks = ceil(jobs / (float)threads);
	warpAffine << <blocks, threads >> > (
		src, src_width * 3, src_height, src_width,
		dst, dst_height, dst_width, 128, d2s, jobs);
}


__global__ void confidenceKernel(float* predict, float* parray, int num_boxes, int num_classes, float conf_threshold, float ratio) {
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= num_boxes) return;

	float* pitem = predict + (4 + num_classes) * position;
	// float objectness = *pitem + 4;
	// if (objectness < conf_threshold) return;

	float* class_confidence = pitem + 4;
	float confidence = *class_confidence++;
	int label = 0;
	for (int i = 1; i < num_classes; i++, class_confidence++) {
		if (*class_confidence > confidence) {
			confidence = *class_confidence;
			label = i;
		}
	}

	// confidence *= objectness;
	if (confidence < conf_threshold) return;

	int index = atomicAdd(parray, 1);
	float x = *pitem++;
	float y = *pitem++;
	float w = *pitem++;
	float h = *pitem++;

	float* parr_pitem = parray + 1 + index * 6;
	*parr_pitem++ = (x - w / 2) * ratio;
	*parr_pitem++ = (y - h / 2) * ratio;
	*parr_pitem++ = (x + w / 2) * ratio;
	*parr_pitem++ = (y + h / 2) * ratio;
	*parr_pitem++ = confidence;
	*parr_pitem++ = label;
}


__device__ float DIOU(
	float ax1, float ay1, float ax2, float ay2,
	float bx1, float by1, float bx2, float by2) {
	// IOU
	float inter_x1 = ax1 > bx1 ? ax1 : bx1;
	float inter_y1 = ay1 > by1 ? ay1 : by1;
	float inter_x2 = ax2 < bx2 ? ax2 : bx2;
	float inter_y2 = ay2 < by2 ? ay2 : by2;
	float inter_lr = inter_x2 > inter_x1 ? inter_x2 - inter_x1 : 0.00f;
	float inter_tb = inter_y2 > inter_y1 ? inter_y2 - inter_y1 : 0.00f;
	float inter = inter_lr * inter_tb;
	float a_area = (ax2 - ax1) * (ay2 - ay1);
	float b_area = (bx2 - bx1) * (by2 - by1);
	float iou = inter / (a_area + b_area - inter);

	// center distance
	float center_a_x = (ax2 - ax1) / 2;
	float center_a_y = (ay2 - ay1) / 2;
	float center_b_x = (bx2 - bx1) / 2;
	float center_b_y = (by2 - by1) / 2;
	float center_distance = (center_a_x - center_b_x) * (center_a_x - center_b_x) + (center_a_y - center_b_y) * (center_a_y - center_b_y);

	// external distance
	float exter_x1 = ax1 < bx1 ? ax1 : bx1;
	float exter_y1 = ay1 < by1 ? ay1 : by1;
	float exter_x2 = ax2 > bx2 ? ax1 : bx2;
	float exter_y2 = ay2 > by2 ? ay2 : by2;
	float exter_distance = (exter_x2 - exter_x1) * (exter_x2 - exter_x1) + (exter_y2 - exter_y1) * (exter_y2 - exter_y1);

	return iou - center_distance / exter_distance;

}


__global__ void NMSKernel(float* boxes, int num_objects, float NMS_threshold) {
	int count = (int)*boxes < num_objects ? (int)*boxes : num_objects;
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= count) return;

	float* pcurrent = boxes + 1 + position * 6;
	for (int i = 0; i < count; i++) {
		float* pitem = boxes + 1 + i * 6;
		if (i == position || pitem[5] != pcurrent[5]) continue;

		if (pitem[4] >= pcurrent[4]) {
			if (pitem[4] == pcurrent[4] && i < position) continue;

			float iou = DIOU(
			pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
			pitem[0],    pitem[1],    pitem[2],    pitem[3]
			);

			if (iou > NMS_threshold) {
				pcurrent[4] = 0;
				return;
			}
		}
	}
}


void postProcess(float* host_input, const int& buffer_size, const int& num_boxes,
	const int& num_classes, const float& conf_threshold, const float& NMS_threshold, const float& ratio,
	const int& num_objects, std::map<int, std::string>& labels, std::vector<YOLO::DetectRet>& detections) {
	
	float* host_output;
	float* dev_input;
	float* dev_output;
	cudaMalloc((void**)&dev_input, buffer_size);
	cudaMalloc((void**)&dev_output, buffer_size);
	int host_size = (num_objects * 6 + 1) * sizeof(float);
	host_output = new float[host_size];

	cudaMemcpy(dev_input, host_input, buffer_size, cudaMemcpyHostToDevice);

	int conf_jobs = num_boxes;
	int conf_threads = 512;
	int conf_blocks = ceil(conf_jobs / (float)conf_threads);
	confidenceKernel << <conf_blocks, conf_threads >> > (dev_input, dev_output, num_boxes, num_classes, conf_threshold, ratio);

	int NMS_jobs = num_objects;
	int NMS_threads = 256;
	int NMS_blocks = ceil(NMS_jobs / (float)NMS_threads);
	NMSKernel << <NMS_blocks, NMS_threads >> > (dev_output, num_objects, NMS_threshold);

	cudaMemcpy(host_output, dev_output, host_size, cudaMemcpyDeviceToHost);

	YOLO::DetectRet box;
	for (int position = 0; position < host_output[0]; position++) {
		float* ret = host_output + 1 + position * 6;
		if (ret[4] == 0) continue;
		box.x1 = ret[0];
		box.y1 = ret[1];
		box.x2 = ret[2];
		box.y2 = ret[3];
		box.conf = ret[4];
		box.name = labels[ret[5]];

		detections.push_back(box);
	}
	
	free(host_output);
	cudaFree(dev_input);
	cudaFree(dev_output);
}
