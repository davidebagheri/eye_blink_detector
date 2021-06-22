#include "eye_blink_detector/eye_blink_detector.h"
#include "yaml-cpp/yaml.h"
#include "eye_blink_detector/utils.h"

int main(int argc, char** argv) {
    // Params
    std::string cfg_path = "../cfg/eb_detector.yaml";
    YAML::Node params = YAML::LoadFile(cfg_path);

    // Eye blink detector
    eb_detector::EyeBlinkDetector eye_blink_detector(params);

    // Video Capture
    int cap_n;
    eb_detector::getParam(params, "video_capture", cap_n, 0);
    cv::VideoCapture cap(cap_n);

    if(!cap.isOpened()){
        std::cerr << "Cannot open the camera." << std::endl;
        return 0;
    }


    cv::Mat frame;

    while (1){
        cap.read(frame);
        eye_blink_detector.detect(frame);
        eye_blink_detector.visualizeResults(&frame);
        cv::imshow("result", frame);

        if((cv::waitKey(2) & 0xFF) == 27)
        break;
    }


    return 0;
}
