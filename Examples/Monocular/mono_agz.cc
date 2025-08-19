/**
* AGZ Zurich MAV Dataset - Pure Monocular SLAM with Pangolin Visualization
* Based on mono_tum.cc but adapted for AGZ dataset format
* 
* Usage: ./mono_agz path_to_vocabulary path_to_settings path_to_agz_data_folder
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<numeric>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_agz path_to_vocabulary path_to_settings path_to_agz_data_folder" << endl;
        cerr << "Example: ./mono_agz ../Vocabulary/ORBvoc.txt ./AGZ_mono.yaml ../../../agz_converted" << endl;
        return 1;
    }

    string vocab_path = argv[1];
    string settings_path = argv[2]; 
    string agz_data_path = argv[3];
    
    cout << "AGZ Zurich MAV Dataset - Pure Monocular SLAM with Pangolin Visualization" << endl;
    cout << "Vocabulary: " << vocab_path << endl;
    cout << "Settings: " << settings_path << endl;
    cout << "AGZ Data: " << agz_data_path << endl << endl;

    // Load images and timestamps
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    
    cout << "Loading images...";
    string image_list_path = agz_data_path + "/images.txt";
    LoadImages(image_list_path, vstrImageFilenames, vTimestamps);
    cout << "LOADED! " << vstrImageFilenames.size() << " images" << endl;

    int nImages = vstrImageFilenames.size();

    if(nImages <= 0)
    {
        cerr << "ERROR: Failed to load images" << endl;
        return 1;
    }

    cout << "Dataset duration: " << (vTimestamps.back() - vTimestamps[0]) << " seconds" << endl << endl;

    // Create SLAM system with Pangolin visualization enabled
    cout << "Initializing ORB-SLAM3 with Pangolin visualization..." << endl;
    ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Starting SLAM processing with visualization..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop - process images with visualization
    for(int ni = 0; ni < nImages; ni++)
    {
        // Read image from file
        string image_filename = vstrImageFilenames[ni];
        string full_image_path = agz_data_path + "/../AGZ/" + image_filename;
        cv::Mat im = cv::imread(full_image_path, cv::IMREAD_UNCHANGED);

        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << full_image_path << endl;
            return 1;
        }

        // Resize image if needed
        if(imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        // Pass the image to the SLAM system - Pangolin will handle visualization
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        SLAM.TrackMonocular(im, tframe);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Display progress with ETA
        if(ni % 500 == 0 || ni < 10)
        {
            float progress = (float)ni / nImages * 100;
            float avg_time = (accumulate(vTimesTrack.begin(), vTimesTrack.begin() + ni + 1, 0.0f) / (ni + 1));
            float eta_minutes = (nImages - ni) * avg_time / 60.0f;
            cout << "Progress: " << ni << "/" << nImages << " (" << fixed << setprecision(1) << progress 
                 << "%) - ETA: " << fixed << setprecision(1) << eta_minutes << " min" << endl;
        }

        // No artificial delays - process at maximum speed for full dataset
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
    cout << "Dataset duration: " << (vTimestamps.back() - vTimestamps[0]) << " seconds" << endl;
    cout << "Average tracking time: " << totaltime / nImages << " seconds" << endl;
    cout << "Processing FPS: " << nImages / totaltime << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("agz_monocular_trajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("agz_monocular_keyframes.txt");
    
    cout << "Trajectory saved to agz_monocular_trajectory.txt and agz_monocular_keyframes.txt" << endl;

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
        iss >> timestamp_str;
        
        string image_path;
        getline(iss, image_path);
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