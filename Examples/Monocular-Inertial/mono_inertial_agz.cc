/**
* This file is part of ORB-SLAM3
*
* AGZ Zurich MAV Dataset - Monocular Inertial SLAM Example
* Based on mono_inertial_euroc.cc
* 
* Usage: ./mono_inertial_agz path_to_vocabulary path_to_settings path_to_agz_data_folder
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>

#include<opencv2/core/core.hpp>

#include<System.h>
#include "ImuTypes.h"

using namespace std;

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps);
void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

double ttrack_tot = 0;

int main(int argc, char *argv[])
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_inertial_agz path_to_vocabulary path_to_settings path_to_agz_data_folder" << endl;
        cerr << "Example: ./mono_inertial_agz ../Vocabulary/ORBvoc.txt ./AGZ.yaml ../../../agz_converted" << endl;
        return 1;
    }

    string vocab_path = argv[1];
    string settings_path = argv[2];
    string agz_data_path = argv[3];
    
    cout << "AGZ Zurich MAV Dataset - ORB-SLAM3 Integration" << endl;
    cout << "Vocabulary: " << vocab_path << endl;
    cout << "Settings: " << settings_path << endl;
    cout << "AGZ Data: " << agz_data_path << endl << endl;

    // Load images and timestamps
    vector<string> vstrImageFilenames;
    vector<double> vTimestampsCam;
    vector<cv::Point3f> vAcc, vGyro;
    vector<double> vTimestampsImu;

    cout << "Loading images...";
    string image_list_path = agz_data_path + "/images.txt";
    LoadImages(image_list_path, vstrImageFilenames, vTimestampsCam);
    cout << "LOADED! " << vstrImageFilenames.size() << " images" << endl;

    cout << "Loading IMU data...";
    string imu_path = agz_data_path + "/imu.txt";
    LoadIMU(imu_path, vTimestampsImu, vAcc, vGyro);
    cout << "LOADED! " << vTimestampsImu.size() << " IMU measurements" << endl;

    int nImages = vstrImageFilenames.size();
    int nImu = vTimestampsImu.size();

    if((nImages <= 0) || (nImu <= 0))
    {
        cerr << "ERROR: Failed to load images or IMU data" << endl;
        return 1;
    }

    // Find first IMU measurement to be considered
    int first_imu = 0;
    while(first_imu < nImu && vTimestampsImu[first_imu] <= vTimestampsCam[0])
        first_imu++;
    first_imu--; // Back up one to get the first IMU measurement before the first image

    cout << "First IMU index: " << first_imu << endl;
    cout << "Dataset duration: " << (vTimestampsCam.back() - vTimestampsCam[0]) << " seconds" << endl << endl;

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout.precision(17);

    // Create SLAM system
    cout << "Initializing ORB-SLAM3..." << endl;
    ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::IMU_MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    // Main loop
    cv::Mat im;
    vector<ORB_SLAM3::IMU::Point> vImuMeas;
    
    cout << "Starting SLAM processing..." << endl;
    
    for(int ni = 0; ni < nImages; ni++)
    {
        // Read image from file
        // Image filename contains "MAV Images/00001.jpg" format
        string image_filename = vstrImageFilenames[ni];
        // Build absolute path: agz_converted/../AGZ/MAV Images/00001.jpg
        string full_image_path = agz_data_path + "/../AGZ/" + image_filename;
        im = cv::imread(full_image_path, cv::IMREAD_UNCHANGED);

        double tframe = vTimestampsCam[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << full_image_path << endl;
            cerr << "Image filename was: '" << image_filename << "'" << endl;
            cerr << "Base path was: '" << agz_data_path << "'" << endl;
            return 1;
        }

        // Resize image if needed
        if(imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
            t_resize += ((double)clock() - (double)clock()) / CLOCKS_PER_SEC;
        }

        // Load IMU measurements between previous and current frame
        vImuMeas.clear();

        if(ni == 0)
            first_imu = max(0, first_imu);

        for(int i = first_imu; i < nImu; i++)
        {
            if(vTimestampsImu[i] <= vTimestampsCam[ni])
            {
                cv::Point3f acc(vAcc[i].x, vAcc[i].y, vAcc[i].z);
                cv::Point3f gyr(vGyro[i].x, vGyro[i].y, vGyro[i].z);
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, vTimestampsImu[i]));
            }
            else
                break;
        }

        if(ni == 0)
            first_imu += vImuMeas.size();

        // Pass the image to the SLAM system
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        SLAM.TrackMonocular(im, tframe, vImuMeas);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        ttrack_tot += ttrack;

        vTimesTrack[ni] = ttrack;

        // Display progress
        if(ni % 100 == 0)
        {
            cout << "Processed " << ni << "/" << nImages << " images. "
                 << "Avg tracking time: " << (ttrack_tot / (ni + 1)) * 1000 << "ms" << endl;
        }

        // Wait to maintain real-time processing if needed
        // Note: Remove this for maximum speed processing
        // usleep(1000000 / 30); // ~30 FPS limit
    }

    cout << endl << "Processing complete!" << endl;

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    
    cout << "-------" << endl << endl;
    cout << "Total Images: " << nImages << endl;
    cout << "Total IMU measurements: " << nImu << endl;
    cout << "Dataset duration: " << (vTimestampsCam.back() - vTimestampsCam[0]) << " seconds" << endl;
    cout << "Average tracking time: " << totaltime / nImages << " seconds" << endl;
    cout << "Frames per second: " << nImages / (vTimestampsCam.back() - vTimestampsCam[0]) << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("agz_trajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("agz_keyframe_trajectory.txt");
    
    cout << "Trajectory saved to agz_trajectory.txt and agz_keyframe_trajectory.txt" << endl;

    return 0;
}

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fImages;
    fImages.open(strImagePath.c_str());
    
    if(!fImages.is_open())
    {
        cerr << "Failed to open image list file: " << strImagePath << endl;
        return;
    }

    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);

    string line;
    while(getline(fImages, line) && !line.empty())
    {
        istringstream iss(line);
        string timestamp_str;
        iss >> timestamp_str;  // Read timestamp first
        
        // Read the rest of the line as image path (can contain spaces)
        string image_path;
        getline(iss, image_path);
        // Trim leading whitespace from image_path
        image_path.erase(0, image_path.find_first_not_of(" \t"));
        
        if(!timestamp_str.empty() && !image_path.empty())
        {
            double timestamp = stod(timestamp_str);
            vTimeStamps.push_back(timestamp);
            vstrImages.push_back(image_path);
        }
    }
    
    fImages.close();
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    
    if(!fImu.is_open())
    {
        cerr << "Failed to open IMU file: " << strImuPath << endl;
        return;
    }

    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    string line;
    while(getline(fImu, line) && !line.empty())
    {
        istringstream iss(line);
        string timestamp_str, gx, gy, gz, ax, ay, az;
        
        if(iss >> timestamp_str >> gx >> gy >> gz >> ax >> ay >> az)
        {
            double timestamp = stod(timestamp_str);
            cv::Point3f gyro(stof(gx), stof(gy), stof(gz));
            cv::Point3f accel(stof(ax), stof(ay), stof(az));
            
            vTimeStamps.push_back(timestamp);
            vGyro.push_back(gyro);
            vAcc.push_back(accel);
        }
    }
    
    fImu.close();
}