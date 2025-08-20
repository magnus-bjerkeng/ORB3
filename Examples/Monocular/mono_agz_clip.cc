/**
* AGZ Zurich MAV Dataset - CLIP-Enhanced Monocular SLAM with Semantic Analysis
* Based on mono_agz.cc with integrated CLIP semantic analysis
* 
* Features:
* - Real-time semantic analysis using CLIP embeddings
* - Dynamic drone color visualization based on text query similarity
* - Frame sampling optimization for performance
* - Inter-process communication with CLIP service
*
* Usage: ./mono_agz_clip path_to_vocabulary path_to_settings path_to_agz_data_folder [text_query]
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<numeric>
#include<iomanip>
#include<thread>
#include<sys/stat.h>
#include<libgen.h>
#include<sstream>

#include<opencv2/core/core.hpp>
#include<opencv2/imgcodecs.hpp>

#include<System.h>

using namespace std;

// CLIP integration configuration
struct CLIPConfig {
    string query = "red car";           // Default semantic query
    int frame_sampling_interval = 8;    // Process every 8 frames (based on CLIP benchmarks)
    string shared_dir = "./clip_shared"; // Directory for SLAM-CLIP communication
    string clip_service_path = "../python/clip_service.py";
    bool enable_clip = true;
    
    // Color mapping based on CLIP similarity thresholds
    struct ColorThresholds {
        float low = 0.15f;      // Green
        float medium = 0.25f;   // Orange/Yellow  
        float high = 0.35f;     // Red
    } thresholds;
};

// CLIP result structure
struct CLIPResult {
    float similarity = 0.0f;
    string color_code = "green";
    string query = "";
    float inference_time = 0.0f;
    int frame_id = 0;
    bool valid = false;
};

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps);
bool CreateDirectory(const string &path);
bool InitializeCLIPCommunication(const CLIPConfig &config);
void SaveImageForCLIP(const cv::Mat &image, int frame_id, const string &shared_dir);
CLIPResult ReadCLIPResult(const string &shared_dir, int frame_id);
void StartCLIPService(const CLIPConfig &config);
void UpdateCLIPQuery(const CLIPConfig &config, const string &new_query);

int main(int argc, char **argv)
{
    if(argc != 4 && argc != 5)
    {
        cerr << endl << "Usage: ./mono_agz_clip path_to_vocabulary path_to_settings path_to_agz_data_folder [text_query]" << endl;
        cerr << "Example: ./mono_agz_clip ../Vocabulary/ORBvoc.txt ./AGZ_mono.yaml ../../../agz_converted \"red car\"" << endl;
        return 1;
    }

    string vocab_path = argv[1];
    string settings_path = argv[2]; 
    string agz_data_path = argv[3];
    
    // CLIP configuration
    CLIPConfig clip_config;
    if(argc == 5) {
        clip_config.query = string(argv[4]);
    }
    
    cout << "AGZ Zurich MAV Dataset - CLIP-Enhanced Monocular SLAM" << endl;
    cout << "Vocabulary: " << vocab_path << endl;
    cout << "Settings: " << settings_path << endl;
    cout << "AGZ Data: " << agz_data_path << endl;
    cout << "CLIP Query: \"" << clip_config.query << "\"" << endl;
    cout << "Frame Sampling: Every " << clip_config.frame_sampling_interval << " frames" << endl << endl;

    // Initialize CLIP communication
    if(clip_config.enable_clip) {
        cout << "Initializing CLIP semantic analysis..." << endl;
        if(!InitializeCLIPCommunication(clip_config)) {
            cerr << "Failed to initialize CLIP communication. Continuing with standard SLAM..." << endl;
            clip_config.enable_clip = false;
        } else {
            cout << "✅ CLIP semantic analysis enabled" << endl;
            
            // Start CLIP service in background
            StartCLIPService(clip_config);
            
            // Give CLIP service time to initialize
            cout << "Waiting for CLIP service to initialize..." << endl;
            this_thread::sleep_for(chrono::seconds(8));
        }
    }

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
    
    // CLIP display will be handled by the Viewer thread to avoid threading conflicts
    // (OpenCV HighGUI is not thread-safe across multiple threads)
    if(clip_config.enable_clip) {
        cout << "✅ CLIP semantic analysis enabled - display handled by Viewer thread" << endl;
        
        // Enable CLIP display in the SLAM system
        SLAM.EnableCLIPDisplay(clip_config.query);
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);
    
    // CLIP processing statistics
    int clip_frames_processed = 0;
    float total_clip_inference_time = 0.0f;

    cout << endl << "-------" << endl;
    cout << "Starting CLIP-enhanced SLAM processing..." << endl;
    cout << "Images in sequence: " << nImages << endl;
    if(clip_config.enable_clip) {
        cout << "CLIP semantic query: \"" << clip_config.query << "\"" << endl;
        cout << "CLIP frame sampling: Every " << clip_config.frame_sampling_interval << " frames" << endl;
    }
    cout << endl;

    // Main loop - process images with CLIP-enhanced visualization
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

        // Store clean RGB image for thread-safe display in Viewer thread
        if (clip_config.enable_clip) {
            // The Viewer thread will handle all OpenCV display operations
            // This eliminates the threading conflict that caused window crashes
            // (OpenCV HighGUI is not thread-safe across multiple threads)
            
            // Pass the clean RGB image to SLAM system for viewer thread display
            SLAM.SetCLIPDisplayData(im, clip_config.query, ni, nImages);
        }

        // CLIP semantic analysis (sample frames for performance)
        CLIPResult clip_result;
        if(clip_config.enable_clip && (ni % clip_config.frame_sampling_interval == 0)) {
            // Save clean RGB image for CLIP processing (without ORB features)
            SaveImageForCLIP(im, ni, clip_config.shared_dir);
            
            // Read CLIP result from previous frame (slight delay for processing)
            if(ni >= clip_config.frame_sampling_interval) {
                int result_frame_id = ni - clip_config.frame_sampling_interval;
                clip_result = ReadCLIPResult(clip_config.shared_dir, result_frame_id);
                if(clip_result.valid) {
                    clip_frames_processed++;
                    total_clip_inference_time += clip_result.inference_time;
                    
                    // Pass CLIP results to SLAM system for thread-safe display in Viewer thread
                    SLAM.SetCLIPResult(clip_result.similarity, clip_result.color_code, clip_result.query);
                }
            }
        }

        // Pass the image to the SLAM system - Pangolin will handle visualization
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        SLAM.TrackMonocular(im, tframe);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Display progress with CLIP analysis info
        if(ni % 500 == 0 || ni < 10 || (clip_result.valid && ni % 50 == 0))
        {
            float progress = (float)ni / nImages * 100;
            float avg_time = (accumulate(vTimesTrack.begin(), vTimesTrack.begin() + ni + 1, 0.0f) / (ni + 1));
            float eta_minutes = (nImages - ni) * avg_time / 60.0f;
            
            cout << "Progress: " << ni << "/" << nImages << " (" << fixed << setprecision(1) << progress << "%)";
            cout << " - ETA: " << fixed << setprecision(1) << eta_minutes << " min";
            
            if(clip_result.valid) {
                cout << " - CLIP: \"" << clip_result.query << "\" = " << fixed << setprecision(3) << clip_result.similarity;
                cout << " (" << clip_result.color_code << ")";
            } else if(clip_config.enable_clip) {
                cout << " - CLIP: processing...";
            }
            cout << endl;
        }

        // No artificial delays - process at maximum speed
    }

    cout << endl << "Processing complete!" << endl;

    // Stop all threads
    SLAM.Shutdown();

    // CLIP display cleanup handled by Viewer thread

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    
    cout << "-------" << endl << endl;
    cout << "📊 PERFORMANCE SUMMARY" << endl;
    cout << "==========================================" << endl;
    cout << "Total Images: " << nImages << endl;
    cout << "Dataset duration: " << (vTimestamps.back() - vTimestamps[0]) << " seconds" << endl;
    cout << "Average tracking time: " << totaltime / nImages << " seconds" << endl;
    cout << "Processing FPS: " << nImages / totaltime << endl;
    
    if(clip_config.enable_clip && clip_frames_processed > 0) {
        cout << endl << "🎯 CLIP SEMANTIC ANALYSIS SUMMARY" << endl;
        cout << "==========================================" << endl;
        cout << "Query: \"" << clip_config.query << "\"" << endl;
        cout << "Frames analyzed: " << clip_frames_processed << " (of " << nImages << ")" << endl;
        cout << "Sampling interval: Every " << clip_config.frame_sampling_interval << " frames" << endl;
        cout << "Average CLIP inference: " << (total_clip_inference_time / clip_frames_processed) << " seconds" << endl;
        cout << "CLIP processing FPS: " << (clip_frames_processed / total_clip_inference_time) << endl;
    }

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("agz_clip_trajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("agz_clip_keyframes.txt");
    
    cout << endl << "Trajectory saved to agz_clip_trajectory.txt and agz_clip_keyframes.txt" << endl;

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

bool CreateDirectory(const string &path) {
    struct stat info;
    if(stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        if(mkdir(path.c_str(), 0755) == 0) {
            return true;
        }
        return false;
    }
    return true; // Directory already exists
}

bool InitializeCLIPCommunication(const CLIPConfig &config) {
    // Create shared directory for SLAM-CLIP communication
    if(!CreateDirectory(config.shared_dir)) {
        cerr << "Failed to create CLIP shared directory: " << config.shared_dir << endl;
        return false;
    }
    
    // Create subdirectories
    CreateDirectory(config.shared_dir + "/images");
    CreateDirectory(config.shared_dir + "/results");
    
    // Write configuration file for CLIP service
    ofstream config_file(config.shared_dir + "/config.txt");
    if(config_file.is_open()) {
        config_file << "query=" << config.query << endl;
        config_file << "frame_sampling=" << config.frame_sampling_interval << endl;
        config_file << "thresholds_low=" << config.thresholds.low << endl;
        config_file << "thresholds_medium=" << config.thresholds.medium << endl;
        config_file << "thresholds_high=" << config.thresholds.high << endl;
        config_file.close();
    }
    
    cout << "CLIP communication initialized: " << config.shared_dir << endl;
    return true;
}

void SaveImageForCLIP(const cv::Mat &image, int frame_id, const string &shared_dir) {
    // Save clean RGB image for CLIP processing
    string image_path = shared_dir + "/images/frame_" + to_string(frame_id) + ".jpg";
    cv::imwrite(image_path, image);
    
    // Create processing request file
    string request_path = shared_dir + "/frame_" + to_string(frame_id) + ".request";
    ofstream request_file(request_path);
    if(request_file.is_open()) {
        request_file << "frame_id=" << frame_id << endl;
        request_file << "image_path=" << image_path << endl;
        request_file << "timestamp=" << chrono::duration_cast<chrono::milliseconds>(
            chrono::system_clock::now().time_since_epoch()).count() << endl;
        request_file.close();
    }
}

CLIPResult ReadCLIPResult(const string &shared_dir, int frame_id) {
    CLIPResult result;
    
    string result_path = shared_dir + "/results/frame_" + to_string(frame_id) + ".json";
    ifstream result_file(result_path);
    
    if(!result_file.is_open()) {
        return result; // Invalid result
    }
    
    // Simple JSON parsing for CLIP results
    string line;
    while(getline(result_file, line)) {
        if(line.find("\"similarity\":") != string::npos) {
            size_t pos = line.find(":") + 1;
            size_t end = line.find(",", pos);
            if(end == string::npos) end = line.find("}", pos);
            if(pos != string::npos && end != string::npos) {
                result.similarity = stof(line.substr(pos, end - pos));
            }
        }
        else if(line.find("\"color_code\":") != string::npos) {
            size_t pos = line.find("\"", line.find(":") + 1) + 1;
            size_t end = line.find("\"", pos);
            if(pos != string::npos && end != string::npos) {
                result.color_code = line.substr(pos, end - pos);
            }
        }
        else if(line.find("\"query\":") != string::npos) {
            size_t pos = line.find("\"", line.find(":") + 1) + 1;
            size_t end = line.find("\"", pos);
            if(pos != string::npos && end != string::npos) {
                result.query = line.substr(pos, end - pos);
            }
        }
        else if(line.find("\"inference_time\":") != string::npos) {
            size_t pos = line.find(":") + 1;
            size_t end = line.find(",", pos);
            if(end == string::npos) end = line.find("}", pos);
            if(pos != string::npos && end != string::npos) {
                result.inference_time = stof(line.substr(pos, end - pos));
            }
        }
    }
    
    result_file.close();
    result.valid = true;
    result.frame_id = frame_id;
    
    // Clean up processed result file
    remove(result_path.c_str());
    
    return result;
}

void StartCLIPService(const CLIPConfig &config) {
    // Start CLIP service as background process
    string command = "python3 " + config.clip_service_path + 
                    " --query \"" + config.query + "\"" +
                    " --shared-dir " + config.shared_dir + " > /dev/null 2>&1 &";
    
    cout << "Starting CLIP service with query: \"" << config.query << "\"" << endl;
    system(command.c_str());
}

void UpdateCLIPQuery(const CLIPConfig &config, const string &new_query) {
    // Update CLIP query by writing to configuration file
    ofstream query_file(config.shared_dir + "/query_update.txt");
    if(query_file.is_open()) {
        query_file << new_query << endl;
        query_file.close();
    }
}