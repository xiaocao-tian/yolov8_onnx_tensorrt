#include "NvInfer.h"
#include <iostream>

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			std::cout << msg << std::endl;
		}
	}
};
