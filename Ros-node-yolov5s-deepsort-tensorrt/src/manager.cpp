#include "manager.hpp"
using std::vector;
using namespace cv;
static Logger gLogger;
Trtyolosort::Trtyolosort()
{}
Trtyolosort::Trtyolosort(string yolo_engine_path,string  sort_engine_path,string detect_model){
	// sort_engine_path_ = sort_engine_path;
	// yolo_engine_path_ = yolo_engine_path;
	if(detect_model=="yolov5"){
		//trt_engine = yolov5_trt_create(yolo_engine_path_);
		cout<<"no model yolov5"<<endl;
	}else {trt_engine = yolov5s_trt_create(yolo_engine_path.c_str());}
	printf("create yolov5-trt , instance = %p\n", trt_engine);
	DS = new DeepSort(sort_engine_path, 128, 256, 0, &gLogger);
}
void Trtyolosort::showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
    cv::Mat temp = img.clone();
	// cv::resize(temp,temp,Size(640,640));
    for (auto box : boxes) {
		if(true)
		{
			cv::Point lt(box.x1, box.y1);
			cv::Point br(box.x2, box.y2);
			//cout<<"********box.weith************"<<endl;
			// cout<<box.height<<endl;
			 cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 2);
			std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
			//std::string lbl = cv::format("ID:%d_C:%d", (int)box.trackID, (int)box.classID);
			//std::string lbl = cv::format("ID:%d_x:%f_y:%f",(int)box.trackID,(box.x1+box.x2)/2,(box.y1+box.y2)/2);
			cv::putText(temp, lbl, lt,cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,255,0));
		}
    }
   cv::imshow("img", temp);
   cv::waitKey(1);
}
int Trtyolosort::TrtDetect(cv::Mat &frame,float &conf_thresh,std::vector<DetectBox> &det,string detect_model){
	// yolo detect
	if(detect_model=="yolov5"){
	//auto ret = yolov5_trt_detect(trt_engine, frame, conf_thresh,det);
			cout<<"no model yolov5"<<endl;
	}else {auto ret = yolov5s_trt_detect(trt_engine, frame, conf_thresh,det);}
	cout<<"****************"<<endl;
	DS->sort(frame,det);
	cout<<"*****11111111111*****"<<endl;
	showDetection(frame,det);
	cout<<"*****222222222222222********"<<endl;
	return 1 ;
	
}
