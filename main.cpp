#include "eye_blink_detector/eye_blink_detector.h"
#include "yaml-cpp/yaml.h"
#include "eye_blink_detector/utils.h"
#include <chrono>

int main(int argc, char** argv) {
    if(argc != 2)
    {
        printf("Usage: %s <camera index>\n", argv[0]);
        return -1;
    }

    // Params
    std::string cfg_path = "../cfg/eb_detector.yaml";
    YAML::Node params = YAML::LoadFile(cfg_path);

    // Eye blink detector
    eb_detector::EyeBlinkDetector eye_blink_detector(params);

    cv::VideoCapture cap;

    // Open camera
    cap.open(argv[1][0] - '0');
    if(!cap.isOpened()){
        std::cerr << "Cannot open the camera." << std::endl;
        return 0;
    }

    if( cap.isOpened()){
        cv::Mat frame;

        while (1){
            auto t1 = std::chrono::high_resolution_clock::now();

            cap.read(frame);


            auto t2 = std::chrono::high_resolution_clock::now();

            eye_blink_detector.detect(frame);

            auto t3 = std::chrono::high_resolution_clock::now();

            eye_blink_detector.visualizeResults(&frame);

            auto t4 = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> cap_read = t2 - t1;
            std::chrono::duration<double, std::milli> eb_det = t3 - t2;
            std::chrono::duration<double, std::milli> vis_res = t4 - t3;
            std::chrono::duration<double, std::milli> tot = t4 - t1;

            std::cout << "cap_read " << cap_read.count() << " eb_det: " << eb_det.count() <<
            " vis_res: " << vis_res.count() << " tot: " << tot.count() << std::endl;

            cv::imshow("result", frame);

            if((cv::waitKey(2) & 0xFF) == 27)
            break;
        }
    }

    return 0;
}
