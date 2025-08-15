#include "dmlModule.h"
#include <iostream>
#include <vector>
#include <windows.h>
#include <chrono>
#include <algorithm>
#include <memory>

#pragma warning(disable:4996)

#define CHECKORT(x,y) if (!CheckStatus(x,y)) return false;

/**
 * @brief 字符串转换为宽字符串
 * @param s 输入字符串
 * @return 宽字符串
 */
std::wstring String2WString(const std::string& s) {
    // 获取所需缓冲区大小
    size_t nDestSize = ::MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, NULL, 0);
    if (nDestSize == 0) {
        return std::wstring();
    }
    
    // 分配缓冲区并转换
    std::unique_ptr<wchar_t[]> wchDest(new wchar_t[nDestSize]);
    ::MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, wchDest.get(), static_cast<int>(nDestSize));
    
    return std::wstring(wchDest.get());
}

/**
 * @brief 字符串转换为UTF-8编码
 * @param str 输入字符串
 * @return UTF-8编码字符串
 */
std::string StringToUTF8(const std::string& str) {
    // 获取宽字符所需缓冲区大小
    size_t nwLen = ::MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, NULL, 0);
    if (nwLen == 0) {
        return std::string();
    }
    
    // 分配宽字符缓冲区并转换
    std::unique_ptr<wchar_t[]> pwBuf(new wchar_t[nwLen]);
    ::MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, pwBuf.get(), static_cast<int>(nwLen));
    
    // 获取UTF-8所需缓冲区大小
    size_t nLen = ::WideCharToMultiByte(CP_UTF8, 0, pwBuf.get(), -1, NULL, 0, NULL, NULL);
    if (nLen == 0) {
        return std::string();
    }
    
    // 分配UTF-8缓冲区并转换
    std::unique_ptr<char[]> pBuf(new char[nLen]);
    ::WideCharToMultiByte(CP_UTF8, 0, pwBuf.get(), -1, pBuf.get(), static_cast<int>(nLen), NULL, NULL);
    
    return std::string(pBuf.get());
}

/**
 * @brief 转换UTF-8编码并获取宽字符串
 * @param onnx_path ONNX模型路径
 * @return 宽字符串
 */
static std::wstring convertU8andGetWstr(const char* onnx_path) {
    if (!onnx_path) {
        return std::wstring();
    }
    return String2WString(StringToUTF8(std::string(onnx_path)));
}

/**
 * @brief 构造函数
 */
IDML::IDML() 
    : input_tensor_size(1), 
      m_env(nullptr), 
      m_session_options(nullptr), 
      m_session(nullptr), 
      m_memory_info(nullptr), 
      m_allocator(nullptr), 
      m_input_tensor(nullptr), 
      m_output_tensor(nullptr), 
      m_ort_api(OrtGetApiBase()->GetApi(ORT_API_VERSION)),
      m_input_name("images"), 
      m_output_name("output"),
      out1(0), 
      out2(0), 
      imgsize(0),
      total_pixels_count(0),
      f1(1.f / 255.0f) {
}

/**
 * @brief 析构函数
 */
IDML::~IDML() {
    // 释放ONNX Runtime资源
    if (m_env) m_ort_api->ReleaseEnv(m_env);
    if (m_memory_info) m_ort_api->ReleaseMemoryInfo(m_memory_info);
    if (m_session) m_ort_api->ReleaseSession(m_session);
    if (m_session_options) m_ort_api->ReleaseSessionOptions(m_session_options);
    if (m_input_tensor) m_ort_api->ReleaseValue(m_input_tensor);
    if (m_output_tensor) m_ort_api->ReleaseValue(m_output_tensor);
    if (m_allocator) m_ort_api->ReleaseAllocator(m_allocator);
}

/**
 * @brief 检查ONNX Runtime状态
 */
bool IDML::CheckStatus(OrtStatus* status, int line) {
    if (status != NULL) {
        const char* msg = m_ort_api->GetErrorMessage(status);
        std::cout << msg << " of " << line << std::endl;
        m_ort_api->ReleaseStatus(status);
        return false;
    }
    return true;
}

/**
 * @brief 初始化接口
 */
bool IDML::InitInterface(const wchar_t* onnx_path) {
    CHECKORT(m_ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "SuperResolutionA", &m_env), __LINE__);
    CHECKORT(m_ort_api->CreateSessionOptions(&m_session_options), __LINE__);
    CHECKORT(m_ort_api->SetSessionGraphOptimizationLevel(m_session_options, ORT_ENABLE_BASIC), __LINE__);
    CHECKORT(m_ort_api->DisableMemPattern(m_session_options), __LINE__);
    m_ort_api->SetSessionExecutionMode(m_session_options, ORT_SEQUENTIAL);

    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_DML(m_session_options, 0);
    
    CHECKORT(m_ort_api->CreateSession(m_env, onnx_path, m_session_options, &m_session), __LINE__);

    // 解析模型
    return parseModelInfo();
}

/**
 * @brief 解析输入信息
 */
bool IDML::parseInput() {
    size_t input_num = 0;
    size_t shape_size = 0;
    char* input_names_temp = nullptr;
    OrtTypeInfo* input_typeinfo = nullptr;
    const OrtTensorTypeAndShapeInfo* Input_tensor_info = nullptr;

    // 获取输入节点数量
    CHECKORT(m_ort_api->SessionGetInputCount(m_session, &input_num), __LINE__);

    // 遍历输入节点索引
    for (size_t i = 0; i < input_num; i++) {
        CHECKORT(m_ort_api->SessionGetInputName(m_session, i, m_allocator, &input_names_temp), __LINE__);

        if (strcmp(m_input_name, input_names_temp)) {
            continue;
        }

        // 获取维度信息
        CHECKORT(m_ort_api->SessionGetInputTypeInfo(m_session, i, &input_typeinfo), __LINE__);
        CHECKORT(m_ort_api->CastTypeInfoToTensorInfo(input_typeinfo, &Input_tensor_info), __LINE__);
        CHECKORT(m_ort_api->GetDimensionsCount(Input_tensor_info, &shape_size), __LINE__);

        std::vector<int64_t> temp(shape_size);
        CHECKORT(m_ort_api->GetDimensions(Input_tensor_info, temp.data(), shape_size), __LINE__);

        if (temp.empty()) {
            std::cout << "获取输入数据失败" << std::endl;
            return false;
        }

        if (temp[2] != temp[3]) {
            std::cout << "不支持输入(image size)不对称的模型" << std::endl;
            return false;
        }

        // 处理维度
        input_tensor_size = 1;
        for (size_t j = 0; j < shape_size; j++) {
            input_tensor_size *= temp[j];
        }

        m_input_dims = std::move(temp);
        imgsize = static_cast<int>(temp[3]);
        
        // 重新分配blob内存
        blob = std::make_unique<float[]>(imgsize * imgsize * 3);
        total_pixels_count = imgsize * imgsize;
        
        std::cout << "获取输入Done. Shape is  " << temp[0] << "_" << temp[1] << "_" << temp[2] << "_" << temp[3] << std::endl;
        return true;
    }

    return false;
}

/**
 * @brief 解析输出信息
 */
bool IDML::parseOutput() {
    size_t shape_size = 0;
    size_t num_output_nodes = 0;
    char* output_names_temp = nullptr;
    OrtTypeInfo* output_typeinfo = nullptr;
    const OrtTensorTypeAndShapeInfo* output_tensor_info = nullptr;

    CHECKORT(m_ort_api->SessionGetOutputCount(m_session, &num_output_nodes), __LINE__);

    for (size_t i = 0; i < num_output_nodes; i++) {
        // 获取输出名
        CHECKORT(m_ort_api->SessionGetOutputName(m_session, i, m_allocator, &output_names_temp), __LINE__);
        if (strcmp(m_output_name, output_names_temp)) {
            continue;
        }

        CHECKORT(m_ort_api->SessionGetOutputTypeInfo(m_session, i, &output_typeinfo), __LINE__);
        CHECKORT(m_ort_api->CastTypeInfoToTensorInfo(output_typeinfo, &output_tensor_info), __LINE__);
        CHECKORT(m_ort_api->GetDimensionsCount(output_tensor_info, &shape_size), __LINE__);

        std::vector<int64_t> temp(shape_size);
        CHECKORT(m_ort_api->GetDimensions(output_tensor_info, temp.data(), shape_size), __LINE__);

        if (temp.empty()) {
            std::cout << "获取输出数据失败" << std::endl;
            return false;
        }

        std::cout << "获取输出Done. Shape is  " << temp[0] << "_" << temp[1] << "_" << temp[2] << std::endl;
        out1 = static_cast<int>(temp[1]);
        out2 = static_cast<int>(temp[2]);
        m_output_dims = std::move(temp);
        return true;
    }

    // 未找到
    return false;
}

/**
 * @brief 解析模型信息
 */
bool IDML::parseModelInfo() {
    // 获取模型信息
    CHECKORT(m_ort_api->GetAllocatorWithDefaultOptions(&m_allocator), __LINE__);
    
    // 解析输入
    if (!parseInput()) {
        return false;
    }
    
    // 解析输出
    if (!parseOutput()) {
        return false;
    }

    // 创建Host内存信息（只需创建一次）
    CHECKORT(m_ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m_memory_info), __LINE__);

    return true;
}

/**
 * @brief 解析模型文件
 */
bool IDML::AnalyticalModel(const char* onnx_path) {
    if (!onnx_path) {
        return false;
    }
    return InitInterface(convertU8andGetWstr(onnx_path).c_str());
}

/**
 * @brief 解析模型文件
 */
bool IDML::AnalyticalModel(const std::string& onnx_path) {
    return InitInterface(convertU8andGetWstr(onnx_path.c_str()).c_str());
}

/**
 * @brief 释放资源
 */
void IDML::Release() {
    // 析构函数会自动释放资源
    delete this;
}

/**
 * @brief 执行目标检测
 */
float* IDML::Detect(BYTE* img) {
    if (!img || !blob) {
        return nullptr;
    }

    // 图像数据预处理
    for (int i = 0; i < total_pixels_count; i++) {
        int g_idx = i + total_pixels_count;
        int b_idx = i + total_pixels_count * 2;

        blob[i] = img[i * 3 + 2] * f1;      // R
        blob[g_idx] = img[i * 3 + 1] * f1;  // G
        blob[b_idx] = img[i * 3] * f1;      // B
    }
    
    // 创建输入tensor（如果尚未创建）
    if (!m_input_tensor) {
        CHECKORT(m_ort_api->CreateTensorWithDataAsOrtValue(
            m_memory_info, 
            blob.get(), 
            input_tensor_size * sizeof(float), 
            m_input_dims.data(), 
            m_input_dims.size(), 
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
            &m_input_tensor), __LINE__);
    } else {
        // 更新tensor数据
        CHECKORT(m_ort_api->FillTensorWithData(
            m_input_tensor,
            m_memory_info,
            blob.get(),
            input_tensor_size * sizeof(float)), __LINE__);
    }

    // 执行推理
    CHECKORT(m_ort_api->Run(
        m_session, 
        NULL, 
        &m_input_name, 
        &m_input_tensor, 
        1, 
        &m_output_name, 
        1, 
        &m_output_tensor), __LINE__);
        
    // 获取输出数据
    void* output_data = nullptr;
    CHECKORT(m_ort_api->GetTensorMutableData(m_output_tensor, &output_data), __LINE__);
    
    // 注意：这里返回的是内部数据指针，调用者不应释放它
    return static_cast<float*>(output_data);
}
