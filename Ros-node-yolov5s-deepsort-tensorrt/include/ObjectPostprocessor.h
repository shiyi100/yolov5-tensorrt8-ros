#include<Eigen/Dense>
#include <Eigen/Core>
#include<vector>
#include<iostream>
#include <numeric>
#include<map>
#include <cmath>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;
struct track_info
{
    int track_times=0;
    int miss_time=0;
    int track_id;
    string clas;
    vector<float> orientation;
    vector<float>x_p;//位置
    vector<float>y_p;
    vector<float>x_v;//速度
    vector<float>y_v;
    vector<float> time_stamp;
};

class ObjectPostprocessor {
    public:
    // std::map<string ,Eigen::Matrix3f>intrinsic_map;
    // std::map<string ,Eigen::Matrix4f>extrinsic_map;Eigen::Matrix<float, 2, 3>
    Eigen::Matrix<float, 3, 3> intrinsic_mtx;
    Eigen::Matrix<float, 4, 4> extrinsic_mtx;
    //EigenMap<std::string, Eigen::Matrix3d> K_;
    ObjectPostprocessor()
    {
        cout<<"ObjectPostprocessor init"<<endl;
    }
    ~ObjectPostprocessor()
    {
        cout<<"ObjectPostprocessor destory"<<endl;
    }
    // intrinsic camera parameter
    // @brief: evaluating y value of given x for a third-order polynomial function
    bool  Process2D(Eigen::Matrix<float ,4,1>uv_points,
    Eigen::Matrix<float ,2,1>&xy_points,float &width); 
    bool calculate_param(Eigen::Matrix<float ,2,1>&xy_points,int track_id,
                        float v_x,float v_y,float orientation);
    float calculate(float distance,float time,float v);
    // convert image point to the camera coordinate
    // & fit the line using polynomial
    void Process3D();
    void GetIm2CarHomography();
    void filteredYellow(const Mat &inputImage, int& pix_number);
    void filteredRed(const Mat &inputImage, int& pix_number);
    void filteredGreen(const Mat &inputImage, int& pix_number);
    // @brief: fit camera lane line using polynomial
    // minimum number to fit a curve
    // number of lane type (13)
    std::map<int,track_info>track_map; 
    private:
    // std::map<string,Eigen::Matrix<float, 3, 3>> homography_ground2image_;
    // std::map<string,Eigen::Matrix<float, 3, 3>> homography_image2ground_;
    Eigen::Matrix<float, 3, 3> homography_ground2image_;
    Eigen::Matrix<float, 3, 3> homography_image2ground_;
    // xy points for the ground plane, uv points for image plane
};

















