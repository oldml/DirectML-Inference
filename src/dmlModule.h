#pragma once

#include <dml/dml_provider_factory.h>
#include <dml/cpu_provider_factory.h>
#include <vector>
#include <iostream>
#include <string>
#include <memory>

/**
 * @brief DirectML推理类
 */
class IDML {
public:
    /**
     * @brief 构造函数
     */
    IDML();

    /**
     * @brief 析构函数
     */
    ~IDML();

    /**
     * @brief 解析模型文件
     * @param onnx_path ONNX模型文件路径
     * @return 是否成功
     */
    bool AnalyticalModel(const char* onnx_path);

    /**
     * @brief 解析模型文件
     * @param onnx_path ONNX模型文件路径
     * @return 是否成功
     */
    bool AnalyticalModel(const std::string& onnx_path);

    /**
     * @brief 执行目标检测
     * @param img 输入图像数据
     * @return 检测结果
     */
    float* Detect(BYTE* img);

    /**
     * @brief 释放资源
     */
    void Release();

private:
    size_t input_tensor_size;                           //! 输入数据大小
    std::unique_ptr<float[]> floatarr;                  //! 输出数据数组
    OrtEnv* m_env;                                      //! ONNX运行环境
    OrtSessionOptions* m_session_options;              //! 会话配置
    OrtSession* m_session;                             //! 会话对象
    OrtMemoryInfo* m_memory_info;                      //! 内存信息
    OrtAllocator* m_allocator;                         //! 分配器
    OrtValue* m_input_tensor;                          //! 输入tensor
    OrtValue* m_output_tensor;                         //! 输出tensor
    const OrtApi* m_ort_api;                           //! ONNX Runtime API

    // 模型输入输出信息
    std::vector<int64_t> m_input_dims;                 //! 输入维度
    std::vector<int64_t> m_output_dims;                //! 输出维度
    const char* m_input_name;                          //! 输入名称
    const char* m_output_name;                         //! 输出名称

    // 图像处理相关
    std::unique_ptr<float[]> blob;                     //! 图像数据缓冲区
    int total_pixels_count;                            //! 总像素数
    const float f1;                                    //! 归一化系数 (1/255.0)

    /**
     * @brief 检查ONNX Runtime状态
     * @param status 状态对象
     * @param line 行号
     * @return 是否成功
     */
    bool CheckStatus(OrtStatus* status, int line);

    /**
     * @brief 解析输入信息
     * @return 是否成功
     */
    bool parseInput();

    /**
     * @brief 解析输出信息
     * @return 是否成功
     */
    bool parseOutput();

    /**
     * @brief 解析模型信息
     * @return 是否成功
     */
    bool parseModelInfo();

    /**
     * @brief 初始化接口
     * @param onnx_path ONNX模型文件路径
     * @return 是否成功
     */
    bool InitInterface(const wchar_t* onnx_path);

public:
    int out1;                                          //! 输出维度1
    int out2;                                          //! 输出维度2
    int imgsize;                                       //! 图像尺寸
};
