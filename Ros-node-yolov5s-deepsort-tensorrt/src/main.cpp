#include<iostream>
#include "manager.hpp"
#include<Eigen/Core>
#include<Eigen/Dense>
#include<opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>
#include <vector>
#include <chrono>
#include <map>
#include <cmath>
#include "string"
#include <time.h>
#include <cv_bridge/cv_bridge.h>
#include"std_msgs/Header.h"
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/fill_image.h>
#include <signal.h>
#include <std_srvs/Empty.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sstream>
//#include "MvGmslCamera.h"
#include "opencv2/imgproc/imgproc.hpp"
#include<ObjectPostprocessor.h>
#include <can_control_msgs/autocontrol.h>
#include<map>
using namespace std;
using namespace cv;

std::map<std::string,float>heightDict;
enum class_enmu {person=1,
            rider=2,
            car=3,
            bus=4,
            truck=5,
            bike=6,
            motor=7,
            tl_green=8,
            tl_red=9,
            tl_yellow=10,
            tl_none=11,
            traffic_sign=12,
            train=13};
template <typename T>
vector<size_t> sort_indexes_e(vector<T> &v)
{
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
    return idx;
}
struct calibrate_matrix
{
    Mat project_mtx;
    Mat mtx;
    Mat dist;
    Size shape;

};

class camera_objects
{
private:
    ros::Publisher pub_objs;
    ros::Publisher pub_dect_img;
    ros::Subscriber sub;
	float conf_thre = 0.4;
	string detetct_modle_name="yolov5s";
	calibrate_matrix calibrate_data;
    ObjectPostprocessor Objectprocess;
    void imageCallback(const sensor_msgs::ImageConstPtr &img);
	//void get_calibrate_matrix(string calibration_file);
	ObjectPostprocessor object_process;
public:
	Trtyolosort yosort;
    camera_objects(ros::NodeHandle& detection_nod);
    ~camera_objects()
    {
        cout<<"destory camera_objects"<<endl;
    }
};
camera_objects::camera_objects(ros::NodeHandle& detection_nod)
{
	pub_objs = detection_nod.advertise<can_control_msgs::autocontrol>("/camera/fusion", 10);
    pub_dect_img= detection_nod.advertise<sensor_msgs::Image>("/camera/dect_vis", 10);
    sub = detection_nod.subscribe<sensor_msgs::Image>("/camera/image", 10, &camera_objects::imageCallback, this);
	Eigen::Matrix<float, 3, 3> intrinsic_params_inverse;
    intrinsic_params_inverse<<1.0526815585362938e+03, 0., 2.6849938138667767e+02, 
                                                            0.,1.0403057567975313e+03, 2.3913284548103204e+02,
                                                             0., 0., 1;
    Eigen::Matrix<float, 4, 4> CameraExtrinsicMat;
    CameraExtrinsicMat<<1., 0., 0., 0. , 
                                                    0., 1., 0., 0., 
                                                    0., 0., 1., 0.,
                                                     0., 0., 0.,1. ;
    // cv2eigen(calibrate_data.mtx, intrinsic_params_inverse);
    // cv2eigen(calibrate_data.project_mtx, CameraExtrinsicMat);
    Objectprocess.intrinsic_mtx= intrinsic_params_inverse;
    Objectprocess.extrinsic_mtx = CameraExtrinsicMat;
    Objectprocess.GetIm2CarHomography();
}
// void get_calibrate_matrix(string calibration_file)
// {
//     calibrate_matrix calibrate_data;
//     cv::FileStorage fs;
//     fs.open("/home/nvidia/yolov5s_deepsort/src/yolov5s-deepsort-tensorrt-main/calibrate_file.yaml",cv::FileStorage::READ);
//     if (!fs.isOpened())
//     {
//         ROS_ERROR("[%s] Cannot open file calibration file '%s'", calibration_file.c_str());
//         ros::shutdown();
//         return;
//     }
//     //fs["CameraExtrinsicMat"] >> data.CameraExtrinsicMat;
//     fs["CameraMat"] >> calibrate_data.mtx;
//     fs["CameraExtrinsicMat"] >> calibrate_data.project_mtx;
//     //fs["ImageSize"] >> data.ImageSize;
//     fs["DistCoeff"] >> calibrate_data.dist;
// }


void camera_objects::imageCallback(const sensor_msgs::ImageConstPtr &img)
{
	cv_bridge::CvImagePtr cv_ptr; //申明一个CvImagePtr
    try
    {
        cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    Mat frame = cv_ptr->image;
	std::vector<DetectBox> det;
	auto start = std::chrono::system_clock::now();
	yosort.TrtDetect(frame,conf_thre,det,detetct_modle_name);
	auto end = std::chrono::system_clock::now();
	int delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
	std::cout  << "delay_infer:" << delay_infer << "ms" << std::endl;
    //rgblight 
    can_control_msgs::autocontrol  camera_control;
    float front_min_dis=0;
    float right_min_dis=0;
    float left_min_dis=0;
    std::vector<DetectBox> traffic_lights;
    std::vector<int> traffic_lights_x;
    cout<<"`122222111)"<<endl;
    for(int i=0;i<det.size();i++)
    {
        if(det[i].classID==8 || det[i].classID==9 || det[i].classID==10 || det[i].classID==11){
            if(150<=(det[i].x1+det[i].x2)/2<=450 && (det[i].y1+det[i].y2)/2<=200)
                traffic_lights.push_back(det[i]);
                traffic_lights_x.push_back(int((det[i].x1+det[i].x2)/2));
            continue;
            }
        else if( det[i].classID==12)
            continue;
    Eigen::Matrix<float, 4, 1> uv_points;
    Eigen::Matrix<float, 2, 1> xy_points;
    uv_points << det[i].x1,det[i].y1,det[i].x2,det[i].y2;
    float width_object;
    Objectprocess.Process2D( uv_points, xy_points, width_object);
    float v_x;
    float v_y;
    float orientation;
    Objectprocess.calculate_param(xy_points, det[i].trackID,  v_x,  v_y,  orientation);
    //cout<<"v_x:"<<v_x<<endl;

    camera_control.Front_Class=det[i].classID;
    if(-1.75<xy_points[1]<1.75)
    {
        if(front_min_dis==0){
            front_min_dis=xy_points[0];
            camera_control.Front_L_LongObj=xy_points[0];
            camera_control.Front_L_LatObj=xy_points[1];
            camera_control.Front_V_LongObj=v_x;
			camera_control.Front_V_LatObj=v_y;
             }
        else if(xy_points[0]<front_min_dis){
            front_min_dis=xy_points[0];
            camera_control.Front_L_LongObj=xy_points[0];
            camera_control.Front_L_LatObj=xy_points[1];
            camera_control.Front_V_LongObj=v_x;
			camera_control.Front_V_LatObj=v_y;
        }
    }
    else if(xy_points[1]<-1.75){
        if(left_min_dis==0){
             left_min_dis=sqrt(pow(xy_points[0],2)+pow(xy_points[1],2));
             camera_control.Leftfront_L_LongObj=xy_points[0];
             camera_control.Leftfront_L_LatObj=xy_points[1];
             camera_control.Leftfront_V_LongObj=v_x;
			 camera_control.Leftfront_V_LatObj=v_y;
        }
        else if(sqrt(pow(xy_points[0],2)+pow(xy_points[1],2))<left_min_dis){
            left_min_dis=sqrt(pow(xy_points[0],2)+pow(xy_points[1],2));
            camera_control.Leftfront_L_LongObj=xy_points[0];
            camera_control.Leftfront_L_LatObj=xy_points[1];
            camera_control.Leftfront_V_LongObj=v_x;
			camera_control.Leftfront_V_LatObj=v_y;
        }
    }
    else if(xy_points[1]>1.75){
        if(right_min_dis==0){
             right_min_dis=sqrt(pow(xy_points[0],2)+pow(xy_points[1],2));
             camera_control.Rightfront_L_LongObj=xy_points[0];
              camera_control.Rightfront_L_LatObj=xy_points[1];
             camera_control.Rightfront_V_LongObj=v_x;
			 camera_control.Rightfront_V_LatObj=v_y;
        }
        else if(sqrt(pow(xy_points[0],2)+pow(xy_points[1],2))<right_min_dis){
            right_min_dis=sqrt(pow(xy_points[0],2)+pow(xy_points[1],2));
            camera_control.Rightfront_L_LongObj=xy_points[0];
            camera_control.Rightfront_L_LatObj=xy_points[1];
            camera_control.Rightfront_V_LongObj=v_x;
			camera_control.Rightfront_V_LatObj=v_y;
        }
    }
    }
    vector<size_t> traffic_ids=sort_indexes_e(traffic_lights_x);//从小到大排序
    if(traffic_lights.size()==3 )
    {
        camera_control.left_traffic_light=traffic_lights[traffic_ids[0]].classID-8;  
        camera_control.middle_traffic_light=traffic_lights[traffic_ids[1]].classID-8;
        camera_control.right_traffic_light=traffic_lights[traffic_ids[2]].classID-8;
    }
    else if(traffic_lights.size()==2)
    {
        camera_control.left_traffic_light=traffic_lights[traffic_ids[0]].classID-8;
        camera_control.middle_traffic_light=traffic_lights[traffic_ids[1]].classID-8;
    }
    else if(traffic_lights.size()==1)
        camera_control.middle_traffic_light=traffic_lights[traffic_ids[0]].classID-8;
    camera_control.header.stamp=ros::Time::now();
    //发布图像跟踪投影的六个点
    pub_objs.publish(camera_control);
    //发布图像监测结果
    cv_ptr->image=frame;
    pub_dect_img.publish(cv_ptr->toImageMsg());
    camera_control.left_traffic_light=3;
    camera_control.middle_traffic_light=3;
    camera_control.right_traffic_light=3;
	camera_control.Front_L_LongObj=0;
	camera_control.Front_L_LatObj=0;
	camera_control.Front_V_LongObj=0;
	camera_control.Front_V_LatObj=0;
	camera_control.Front_A_LongObj=0;
	camera_control.Front_A_LongObj=0;
	camera_control.Leftfront_L_LongObj=0;
	camera_control.Leftfront_L_LatObj=0;
	camera_control.Leftfront_V_LongObj=0;
	camera_control.Leftfront_V_LatObj=0;
	camera_control.Leftfront_A_LongObj=0;
	camera_control.Leftfront_A_LongObj=0;
	camera_control.Rightfront_L_LongObj=0;
	camera_control.Rightfront_L_LatObj=0;
	camera_control.Rightfront_V_LongObj=0;
	camera_control.Rightfront_V_LatObj=0;
	camera_control.Rightfront_A_LongObj=0;
	camera_control.Rightfront_A_LongObj=0;
}
int main(int argc, char* argv[]){
    ros::init(argc, argv, "yolov5ssort_node");
    ros::NodeHandle nh;
    heightDict.insert(std::map<string,float>::value_type("car",2));
    heightDict.insert(std::map<string,float>::value_type("bus",3.2));
    heightDict.insert(std::map<string,float>::value_type("rider",0.6));
    heightDict.insert(std::map<string,float>::value_type("motor",0.6));
    heightDict.insert(std::map<string,float>::value_type("person",1.67));
    heightDict.insert(std::map<string,float>::value_type("truck",4));
    heightDict.insert(std::map<string,float>::value_type("bike",0.6));
    heightDict.insert(std::map<string,float>::value_type("train",10));
    string detetct_modle_name="yolov5s";
    // string  yolo_engine ="/home/nvidia/yolov5s_deepsort/src/yolov5s-deepsort-tensorrt-main/yolov5s.engine";
    // string  sort_engine="/home/nvidia/yolov5s_deepsort/src/yolov5s-deepsort-tensorrt-main/deepsort.engine";
    string yolo_engine =nh.param<string>("detect_model","/home/swh/muil_camera_lidar/src/Ros-node-yolov5s-deepsort-tensorrt/yolov5s.engine");
    string sort_engine=nh.param<string>("track_model", "/home/swh/muil_camera_lidar/src/Ros-node-yolov5s-deepsort-tensorrt/deepsort.engine");
    // calibrate_file =nh.param<string>("calibrate_file", "/home/nvidia/yolov5s_deepsort/src/yolov5s-deepsort-tensorrt-main/calibrate_file.yml");
	Trtyolosort yosort(yolo_engine,sort_engine,detetct_modle_name);
	camera_objects  objets(nh);
    objets.yosort=yosort;
    ros::Rate loop_rate(30);
    while(ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();

    }
	
	return 0;
}
