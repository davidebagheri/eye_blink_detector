#ifndef EYE_BLINK_DETECTOR_EYE_BLINK_DETECTOR_H
#define EYE_BLINK_DETECTOR_EYE_BLINK_DETECTOR_H

#include "face_detector.h"
#include "blink_classifier.h"
#include "yaml-cpp/yaml.h"

namespace eb_detector {
    class EyeBlinkDetector {
    public:
        EyeBlinkDetector(const YAML::Node& params);

        bool detect(const cv::Mat& image);

        void visualizeResults(cv::Mat* image);

        const BlinkClassifier& getBlinkClassifier();

        const FaceDetector& getFaceDetector();

    private:
        FaceDetector* face_detector_;

        BlinkClassifier* blink_classifier_;
    };
}

#endif //EYE_BLINK_DETECTOR_EYE_BLINK_DETECTOR_H
