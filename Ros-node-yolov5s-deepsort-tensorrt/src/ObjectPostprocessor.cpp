#include<ObjectPostprocessor.h>


float ObjectPostprocessor::calculate(float distance,float time,float v)
{
    float a=2*(distance-time*v)/time;
    return a*time;
}
bool ObjectPostprocessor::calculate_param(Eigen::Matrix<float ,2,1>&xy_points,int track_id,float v_x,float v_y,float orientation)
{
    auto iter = track_map.find(track_id);
    if(iter != track_map.end())
    {
        if(track_map[track_id].track_times<1)
        {
            track_map[track_id].track_id=track_id;
            track_map[track_id].x_p.push_back(xy_points(0,0));
            track_map[track_id].y_p.push_back(xy_points(1,0));
            track_map[track_id].x_v.push_back(0);
            track_map[track_id].y_v.push_back(0);
            track_map[track_id].track_times++;
            track_map[track_id].time_stamp.push_back(ros::Time::now().toSec());
        }
        else if(0<track_map[track_id].track_times<10)
        {
            track_map[track_id].x_p.push_back(xy_points(0,0));
            track_map[track_id].y_p.push_back(xy_points(1,0));
            //accumulate(v.begin(), v.end(), 0)
            int num=track_map[track_id].x_p.size();
            float t=ros::Time::now().toSec()-track_map[track_id].time_stamp.back();
            float dis_x=track_map[track_id].x_p[num]-track_map[track_id].x_p[num-1];
            float dis_y=track_map[track_id].y_p[num]-track_map[track_id].y_p[num-1];
            float x_v=calculate(dis_x,track_map[track_id].x_v.back(),t);
            float y_v=calculate(dis_y,track_map[track_id].y_v.back(),t);
            track_map[track_id].x_v.push_back(x_v);
            track_map[track_id].y_v.push_back(y_v);
            track_map[track_id].time_stamp.push_back(ros::Time::now().toSec());
            track_map[track_id].track_times++;
            float orientation=atan(y_v/x_v);
            if(abs(orientation-track_map[track_id].orientation.back())<5)
                track_map[track_id].orientation.push_back(track_map[track_id].orientation.back());
            else 
            {
                track_map[track_id].orientation.push_back(orientation);
            }
        }
        else if (track_map[track_id].track_times>10)
        {
            track_map[track_id].x_p.erase(track_map[track_id].x_p.begin());
            track_map[track_id].y_p.erase(track_map[track_id].y_p.begin());
            track_map[track_id].x_v.erase(track_map[track_id].x_v.begin());
            track_map[track_id].y_v.erase(track_map[track_id].y_v.begin());
            track_map[track_id].time_stamp.erase(track_map[track_id].time_stamp.begin());
            track_map[track_id].orientation.erase(track_map[track_id].orientation.begin());
            track_map[track_id].x_p.push_back(xy_points(0,0));
            track_map[track_id].y_p.push_back(xy_points(1,0));
            //accumulate(v.begin(), v.end(), 0)
            int num=track_map[track_id].x_p.size();
            float t=ros::Time::now().toSec()-track_map[track_id].time_stamp.back();
            float dis_x=track_map[track_id].x_p[num]-track_map[track_id].x_p[num-1];
            float dis_y=track_map[track_id].y_p[num]-track_map[track_id].y_p[num-1];
            float x_v=calculate(dis_x,track_map[track_id].x_v.back(),t);
            float y_v=calculate(dis_y,track_map[track_id].y_v.back(),t);
            track_map[track_id].x_v.push_back(x_v);
            track_map[track_id].y_v.push_back(y_v);
            track_map[track_id].time_stamp.push_back(ros::Time::now().toSec());
            track_map[track_id].track_times++;
            float orientation=atan(y_v/x_v);
            if(abs(orientation-track_map[track_id].orientation.back())<5)
                track_map[track_id].orientation.push_back(track_map[track_id].orientation.back());
            else 
            {
                track_map[track_id].orientation.push_back(orientation);
            }
        }
    }
}


bool ObjectPostprocessor::Process2D(Eigen::Matrix<float ,4,1>uv_points,
Eigen::Matrix<float ,2,1>&xy_points,float &width) 
{
    float downleft_x=uv_points[0];
    float downleft_y=uv_points[3];
    float downright_x=uv_points[2];
    float downright_y=uv_points[3];
    float downcenter_x=(downleft_x+downright_x)/2;
    float downcenter_y=(downleft_y+downright_y)/2;
    Eigen::Matrix<float, 3, 1> uv_downcenter;
    uv_downcenter << downcenter_x,downcenter_y,1;
    Eigen::Matrix<float, 3, 1> xy_center;
    xy_center=homography_image2ground_*uv_downcenter;
    Eigen::Matrix<float, 2, 1> xy_center_;
    //if (std::fabs(xy_center(2)) > 2.22507e-308) continue;
    xy_points <<xy_center(0)/xy_center(2),xy_center(1)/xy_center(2);
    //distance=sqrt(xy_center_(0)^2+xy_center_(1)^2);
    Eigen::Matrix<float, 3, 1> uv_downleft;
    uv_downleft << downleft_x,downleft_y,1;
    Eigen::Matrix<float, 3, 1> xy_downleft;
    xy_downleft=homography_image2ground_*uv_downleft;
    Eigen::Matrix<float, 2, 1> xy_downleft_;
    //if (std::fabs(xy_downleft(2)) < 1e-6) continue;
    xy_downleft_ <<xy_downleft(0)/xy_downleft(2),xy_downleft(1)/xy_downleft(2);
    Eigen::Matrix<float, 3, 1> uv_downright;
    uv_downright << downright_x,downright_y,1;
    Eigen::Matrix<float, 3, 1> xy_downright;
    xy_downright=homography_image2ground_*uv_downright;
    Eigen::Matrix<float, 2, 1> xy_downright_;
    //if (std::fabs(xy_downright(2)) < 1e-6) continue;
    xy_downright_ <<xy_downright(0)/xy_downright(2),xy_downright(1)/xy_downright(2);
    width=sqrt(pow((xy_downright_(0)-xy_downleft_(0)),2)+pow((xy_downright_(1)-xy_downleft_(1)),2));
    cout << "Object_postprocess done!"<<endl;
    return true;
}
// convert image point to the camera coordinate
// & fit the line using polynomial
void ObjectPostprocessor::GetIm2CarHomography()
{
    // intrinsic camera parameter
    // K_[camera_name] = intrinsic_map_.at(camera_name).cast<double>();
    //     // Convert degree angles to radian angles
    // double pitch_adj_radian = pitch_adj_degree * degree_to_radian_factor_;
    // double yaw_adj_radian = yaw_adj_degree * degree_to_radian_factor_;
    // double roll_adj_radian = roll_adj_degree * degree_to_radian_factor_;
    //     // We use "right handed ZYX" coordinate system for euler angles
    // // adjust pitch yaw roll in camera coords
    // // Remember that camera coordinate
    // // (Z)----> X
    // //  |
    // //  |
    // //  V
    // //  Y
    // Eigen::Matrix4d Rx;  // pitch
    // Rx << 1, 0, 0, 0, 0, cos(pitch_adj_radian), -sin(pitch_adj_radian), 0, 0,
    //     sin(pitch_adj_radian), cos(pitch_adj_radian), 0, 0, 0, 0, 1;
    // Eigen::Matrix4d Ry;  // yaw
    // Ry << cos(yaw_adj_radian), 0, sin(yaw_adj_radian), 0, 0, 1, 0, 0,
    //     -sin(yaw_adj_radian), 0, cos(yaw_adj_radian), 0, 0, 0, 0, 1;
    // Eigen::Matrix4d Rz;  // roll
    // Rz << cos(roll_adj_radian), -sin(roll_adj_radian), 0, 0, sin(roll_adj_radian),
    //     cos(roll_adj_radian), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

    // adjusted_camera2car_ = ex_camera2car_[camea_name] * Rz * Ry * Rx;
        // Get homography from projection matrix
    // ====
    // Version 1. Direct

    // compute the homography matrix, such that H [u, v, 1]' ~ [X_l, Y_l, 1]
    Eigen::Matrix<float, 3, 3> R = extrinsic_mtx.block(0, 0, 3, 3);
    Eigen::Matrix<float, 3, 1> T = extrinsic_mtx.block(0, 3, 3, 1);
    Eigen::Matrix<float, 3, 3> H;
    Eigen::Matrix<float, 3, 3> H_inv;
    H.block(0, 0, 3, 2) = (intrinsic_mtx * R).block(0, 0, 3, 2);
    H.block(0, 2, 3, 1) = intrinsic_mtx * T;
    H_inv = H.inverse();
    homography_ground2image_= H;
    homography_image2ground_ = H_inv;
}

void ObjectPostprocessor::filteredRed(const Mat &inputImage, int& pix_number){
    pix_number=0;
    Mat hsvImage;
    // resultGray = Mat(hsvImage.rows, hsvImage.cols,CV_8U,cv::Scalar(255));  
    // resultColor = Mat(hsvImage.rows, hsvImage.cols,CV_8UC3,cv::Scalar(255, 255, 255));
    double H=0.0,S=0.0,V=0.0;   
    for(int i=0;i<hsvImage.rows;i++)
    {
        for(int j=0;j<hsvImage.cols;j++)
        {
            H=hsvImage.at<Vec3b>(i,j)[0];
            S=hsvImage.at<Vec3b>(i,j)[1];
            V=hsvImage.at<Vec3b>(i,j)[2];
            if(((0<=H<=10) ||(156<=H<=180))&& (43<=S<=255)&&(46<=V<=255))
            {       
                pix_number++;
            }
        }
    }
}
void ObjectPostprocessor::filteredGreen(const Mat &inputImage, int& pix_number){
    pix_number=0;
    Mat hsvImage;
    // resultGray = Mat(hsvImage.rows, hsvImage.cols,CV_8U,cv::Scalar(255));  
    // resultColor = Mat(hsvImage.rows, hsvImage.cols,CV_8UC3,cv::Scalar(255, 255, 255));
    double H=0.0,S=0.0,V=0.0;   
    for(int i=0;i<hsvImage.rows;i++)
    {
        for(int j=0;j<hsvImage.cols;j++)
        {
            H=hsvImage.at<Vec3b>(i,j)[0];
            S=hsvImage.at<Vec3b>(i,j)[1];
            V=hsvImage.at<Vec3b>(i,j)[2];

            if((35<=H<=77) && (43<=S<=255)&&(46<=V<=255))
            {      
                pix_number++;
            }
        }
    }
}
void ObjectPostprocessor::filteredYellow(const Mat &inputImage, int& pix_number){
    pix_number=0;
    Mat hsvImage;
    // resultGray = Mat(hsvImage.rows, hsvImage.cols,CV_8U,cv::Scalar(255));  
    // resultColor = Mat(hsvImage.rows, hsvImage.cols,CV_8UC3,cv::Scalar(255, 255, 255));
    double H=0.0,S=0.0,V=0.0;   
    for(int i=0;i<hsvImage.rows;i++)
    {
        for(int j=0;j<hsvImage.cols;j++)
        {
            H=hsvImage.at<Vec3b>(i,j)[0];
            S=hsvImage.at<Vec3b>(i,j)[1];
            V=hsvImage.at<Vec3b>(i,j)[2];
            if((26<=H<=34) && (43<=S<=255)&&(46<=V<=255))
            {      
                pix_number++;
            }
        }
    }
}


