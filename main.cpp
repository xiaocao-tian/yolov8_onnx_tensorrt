#include "yolo.h"

int main() {
	std::string engine_file = "../weights/yolov8n.engine";
	std::string class_file  = "../weights/classes.txt";
	float conf_threshold = 0.25;
	float NMS_threshold = 0.45;

	YOLO YOLO(engine_file, class_file, conf_threshold, NMS_threshold);
	YOLO.Inference();

	return 0;
}
