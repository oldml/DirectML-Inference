#pragma once

#include <dml/dml_provider_factory.h>
#include <dml/cpu_provider_factory.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>


class IDML
{
public:
	//  --------------------------- ����API --------------------------- // IStates

	//! ����ģ��
	bool AnalyticalModel(const char* onnx_path);

	//! ����ģ��
	bool AnalyticalModel(const std::string onnx_path);


	float* Detect(BYTE* img);

	//! �ͷ�����
	void Release();

	

private:

	size_t input_tensor_size = 1;						//! �������ݴ�С
	float* floatarr = nullptr;
	OrtEnv* m_env = nullptr;
	OrtSessionOptions* m_session_options = nullptr;
	OrtSession* m_session = nullptr;
	OrtMemoryInfo* m_memory_info = nullptr;
	OrtAllocator* m_allocator = nullptr;
	OrtValue* m_input_tensors = nullptr;		// ����tensor
	OrtValue* m_output_tensors = nullptr;		// ���tensor
	const OrtApi* _ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);	// api


	// Function
	bool CheckStatus(OrtStatus* status, int line);
	bool parseInput();
	bool parseOutput();
	bool parseModelInfo();
	bool InitInterface(const wchar_t* onn_path);

	const char* input_name[1] = { "images" };
	const char* output_name[1] = { "output" };

	std::vector<int64_t> m_input_dims = {};
	std::vector<int64_t> m_output_dims = {};

	float* blob;
	int total_pixels_count ;
	float f1 = 1.f / 255.0f;
	
public:

	int out1 = 0;
	int out2 = 0;
	int imgsize = 0;
};






