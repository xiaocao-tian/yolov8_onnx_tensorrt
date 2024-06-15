#include <fstream>
#include "Logger.hpp"
 
Logger serializeLogger;
void readClassFile(const std::string& class_file, std::map<int, std::string>& labels) {
	std::fstream file(class_file, std::ios::in);
	if (!file.is_open()) {
		std::cout << "Load classes file failed: " << class_file << std::endl;
		system("pause");
		exit(0);
	}
	std::cout << "Load classes file success: " << class_file << std::endl;
	std::string str_line;
	int index = 0;
	while ( getline(file, str_line) ) {
		labels.insert({ index, str_line });
		index++;
	}
	file.close();
}

void readEngineFile(const std::string& ENGINE_FILE, nvinfer1::ICudaEngine*& engine) {
	std::fstream file;
	nvinfer1::IRuntime* runtime;
	file.open(ENGINE_FILE, std::ios::in | std::ios::binary);

	if (!file.is_open()) {
		std::cout << "Load engine file failed: " << ENGINE_FILE << std::endl;
		system("pause");
		exit(0);
	}
	std::cout << "Load engine file success: " << ENGINE_FILE << std::endl;
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	char* model_engine = new char[size];
	file.read(model_engine, size);
	file.close();
	runtime = nvinfer1::createInferRuntime(serializeLogger);
	engine = runtime->deserializeCudaEngine(model_engine, size);
	delete[] model_engine;
}
