#include "dmlModule.h"
#include "cap.h"
#include "nms.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

#define SHOWCV2 1
#define CONFIDENCE 0.4f
#define NMS_THRESHOLD 0.5f

/**
 * @brief 主函数
 * @param argc 参数数量
 * @param argv 参数数组
 * @return 程序退出码
 */
int main(int argc, char* argv[]) {
    try {
        // 检查命令行参数
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <window_title>" << std::endl;
            return -1;
        }

        const char* window_title = argv[1];
        timeBeginPeriod(1);

        // 模型路径
        const char* module_path = "example.onnx";
        
        // 创建DML推理对象
        std::unique_ptr<IDML> frame = std::make_unique<IDML>();
        if (!frame->AnalyticalModel(module_path)) {
            std::cerr << "Failed to load model: " << module_path << std::endl;
            return -1;
        }
        
        int imgsize = frame->imgsize;
        if (imgsize <= 0) {
            std::cerr << "Invalid image size from model" << std::endl;
            return -1;
        }

        // 创建屏幕捕获对象
        std::unique_ptr<capture> c = std::make_unique<capture>(1920, 1080, imgsize, imgsize, window_title);
        
        // 创建OpenCV窗口
#if SHOWCV2
        cv::namedWindow("Detection Result", cv::WINDOW_AUTOSIZE);
#endif

        // 主循环
        while (true) {
            // 捕获屏幕内容
            BYTE* s = static_cast<BYTE*>(c->cap());
            if (!s) {
                std::cerr << "Failed to capture screen" << std::endl;
                continue;
            }

            // 执行目标检测
            float* data = frame->Detect(s);
            if (!data) {
                std::cerr << "Detection failed" << std::endl;
                continue;
            }

            // 生成检测提议框
            std::vector<Box> newbox = generate_yolo_proposals(data, frame->out1, frame->out2, CONFIDENCE, NMS_THRESHOLD);

#if SHOWCV2        
            // 创建OpenCV图像
            cv::Mat a(imgsize, imgsize, CV_8UC3, s);
            
            // 绘制检测框
            for (const Box& detection : newbox) {
                cv::rectangle(a, 
                             cv::Point(static_cast<int>(detection.left), static_cast<int>(detection.top)), 
                             cv::Point(static_cast<int>(detection.right), static_cast<int>(detection.bottom)), 
                             cv::Scalar(0, 255, 0), 1);
            }
            
            // 显示结果
            cv::imshow("Detection Result", a);
            
            // 检查按键退出
            if (cv::waitKey(1) == 27) {  // ESC键退出
                break;
            }
#endif
        }
        
#if SHOWCV2
        // 释放OpenCV窗口资源
        cv::destroyAllWindows();
#endif

        // 释放资源
        timeEndPeriod(1);
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return -1;
    }
}
