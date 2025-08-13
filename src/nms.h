#include<iostream>
#include<vector>
#include<algorithm>
#include <stdio.h>
#include <stdlib.h>
#include "dmlModule.h"
using namespace std;


struct Box
{
	float left, top, right, bottom, confidence;
	int class_label;

	Box() = default;

	Box(float left, float top, float right, float bottom, float confidence, int class_label)
		:left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
};





float iou(const Box& a, const Box& b)
{
	float cleft = max(a.left, b.left);
	float ctop = max(a.top, b.top);
	float cright = min(a.right, b.right);
	float cbottom = min(a.bottom, b.bottom);

	float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
	if (c_area == 0.0f)
		return 0.0f;

	float a_area = max(0.0f, a.right - a.left) * max(0.0f, a.bottom - a.top);
	float b_area = max(0.0f, b.right - b.left) * max(0.0f, b.bottom - b.top);
	return c_area / (a_area + b_area - c_area);
}



vector<Box> generate_yolo_proposals(float* feat_blob, vector<Box>& objects, int out1 ,int out2, float confidence, float  threshold)
{


	for (int boxs_idx = 0; boxs_idx < out1; boxs_idx++)
	{

		const int basic_pos = boxs_idx * out2;

		float box_objectness = feat_blob[basic_pos + 4];
		if (box_objectness > confidence)
		{

			float x_center = feat_blob[basic_pos + 0];
			float y_center = feat_blob[basic_pos + 1];
			float w = feat_blob[basic_pos + 2];
			float h = feat_blob[basic_pos + 3];

			float x0 = x_center - w * 0.5f;
			float y0 = y_center - h * 0.5f;
			float x1 = x_center + w * 0.5f;
			float y1 = y_center + h * 0.5f;


			for (int class_idx = 5; class_idx < out2; class_idx++)
			{

				float box_prob = feat_blob[basic_pos + class_idx];

				if (box_prob > confidence)
				{

					Box a;
					a.left = x0;
					a.top = y0;
					a.right = x1;
					a.bottom = y1;
					a.class_label = class_idx - 5;
					a.confidence = box_prob;
					objects.emplace_back(a);

				}

			}
		}
	}
	std::sort(objects.begin(), objects.end(), [](vector<Box>::const_reference a, vector<Box>::const_reference b)
		{
			return a.confidence > b.confidence;
		});

	vector<Box> output;
	output.reserve(objects.size());

	vector<bool> remove_flags(objects.size());


	for (int i = 0; i < objects.size(); ++i) {

		if (remove_flags[i]) continue;

		auto& a = objects[i];
		output.emplace_back(a);


		for (int j = i + 1; j < objects.size(); ++j) {
			if (remove_flags[j]) continue;

			auto& b = objects[j];
			if (b.class_label == a.class_label) {
				if (iou(a, b) >= threshold)
					remove_flags[j] = true;
			}
		}
	}

	return output;

}
