#include "eye_blink_detector/eye_blink_detector.h"
#include "yaml-cpp/yaml.h"
#include "eye_blink_detector/utils.h"

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
            cap.read(frame);

            eye_blink_detector.detect(frame);

            eye_blink_detector.visualizeResults(&frame);

            cv::imshow("result", frame);

            if((cv::waitKey(2) & 0xFF) == 27)
            break;
        }
    }

    return 0;
}
