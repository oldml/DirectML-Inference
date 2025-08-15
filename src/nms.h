#ifndef NMS_H
#define NMS_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "dmlModule.h"

/**
 * @brief 检测框结构体
 */
struct Box {
    float left, top, right, bottom, confidence;
    int class_label;

    // 默认构造函数
    Box() : left(0), top(0), right(0), bottom(0), confidence(0), class_label(0) {}

    // 带参数的构造函数
    Box(float l, float t, float r, float b, float conf, int label)
        : left(l), top(t), right(r), bottom(b), confidence(conf), class_label(label) {}

    // 拷贝构造函数
    Box(const Box& other)
        : left(other.left), top(other.top), right(other.right), 
          bottom(other.bottom), confidence(other.confidence), class_label(other.class_label) {}

    // 赋值操作符
    Box& operator=(const Box& other) {
        if (this != &other) {
            left = other.left;
            top = other.top;
            right = other.right;
            bottom = other.bottom;
            confidence = other.confidence;
            class_label = other.class_label;
        }
        return *this;
    }
};

/**
 * @brief 计算两个检测框的交并比(IOU)
 * @param a 第一个检测框
 * @param b 第二个检测框
 * @return 交并比值
 */
float iou(const Box& a, const Box& b) {
    float cleft = std::max(a.left, b.left);
    float ctop = std::max(a.top, b.top);
    float cright = std::min(a.right, b.right);
    float cbottom = std::min(a.bottom, b.bottom);

    float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top);
    float b_area = std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top);
    return c_area / (a_area + b_area - c_area);
}

/**
 * @brief 生成YOLO检测提议框
 * @param feat_blob 特征数据
 * @param out1 输出维度1
 * @param out2 输出维度2
 * @param confidence_threshold 置信度阈值
 * @param nms_threshold NMS阈值
 * @return 检测框向量
 */
std::vector<Box> generate_yolo_proposals(const float* feat_blob, int out1, int out2, 
                                         float confidence_threshold, float nms_threshold) {
    // 输入参数有效性检查
    if (!feat_blob || out1 <= 0 || out2 <= 0 || confidence_threshold < 0 || nms_threshold < 0) {
        return std::vector<Box>();
    }

    std::vector<Box> objects;
    
    for (int boxs_idx = 0; boxs_idx < out1; boxs_idx++) {
        const int basic_pos = boxs_idx * out2;
        float box_objectness = feat_blob[basic_pos + 4];
        
        if (box_objectness > confidence_threshold) {
            float x_center = feat_blob[basic_pos + 0];
            float y_center = feat_blob[basic_pos + 1];
            float w = feat_blob[basic_pos + 2];
            float h = feat_blob[basic_pos + 3];

            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;
            float x1 = x_center + w * 0.5f;
            float y1 = y_center + h * 0.5f;

            for (int class_idx = 5; class_idx < out2; class_idx++) {
                float box_prob = feat_blob[basic_pos + class_idx];
                
                if (box_prob > confidence_threshold) {
                    Box a(x0, y0, x1, y1, box_prob, class_idx - 5);
                    objects.emplace_back(a);
                }
            }
        }
    }
    
    // 按置信度降序排序
    std::sort(objects.begin(), objects.end(), [](const Box& a, const Box& b) {
        return a.confidence > b.confidence;
    });

    // 应用非极大值抑制
    std::vector<Box> output;
    output.reserve(objects.size());
    
    std::vector<bool> remove_flags(objects.size(), false);

    for (size_t i = 0; i < objects.size(); ++i) {
        if (remove_flags[i]) continue;
        
        const Box& a = objects[i];
        output.emplace_back(a);

        for (size_t j = i + 1; j < objects.size(); ++j) {
            if (remove_flags[j]) continue;
            
            const Box& b = objects[j];
            if (b.class_label == a.class_label) {
                if (iou(a, b) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }

    return output;
}

#endif // NMS_H
