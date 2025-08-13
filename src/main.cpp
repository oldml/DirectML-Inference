#include "dml-module.h"
#include "cap.h"
#include "nms.h"
#include <opencv2/opencv.hpp>
#define SHOWCV2 1
#define CONFIDENCE 0.4f
#define NMS_THRESHOLD 0.5f

int main()
{
	timeBeginPeriod(1);

	const char* module_path = "1.onnx";
	auto* frame = new IDML();
	frame->AnalyticalModel(module_path);
	int imgsize = frame->imgsize;

	capture c(1920, 1080, imgsize, imgsize, "CrossFire");

	while (true)
	{
		BYTE* s = (BYTE*)c.cap();

		float* data = frame->Detect(s);

		vector<Box> oldbox;
		vector<Box> newbox = generate_yolo_proposals(data, oldbox, frame->out1, frame->out2, CONFIDENCE, NMS_THRESHOLD);

#if SHOWCV2        
		cv::Mat a = cv::Mat(imgsize, imgsize, CV_8UC3, s);
#endif 

		// 绘制检测框
		for (const Box& detection : newbox)
		{
#if SHOWCV2        
			cv::rectangle(a, cv::Point((int)detection.left, (int)detection.top), cv::Point((int)detection.right, (int)detection.bottom), cv::Scalar(0, 255, 0), 1);
#endif 
		}

#if SHOWCV2
		cv::imshow("c", a);
		cv::waitKey(1);
#endif
	}
}
