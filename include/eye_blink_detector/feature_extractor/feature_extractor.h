#ifndef EYE_BLINK_DETECTOR_FEATURE_EXTRACTOR_H
#define EYE_BLINK_DETECTOR_FEATURE_EXTRACTOR_H

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "eye_blink_detector/utils.h"

namespace eb_detector {
    class FeatureExtractor {
    public:
        FeatureExtractor(const YAML::Node& params){}

        virtual cv::Mat forward(const std::vector<cv::Mat>& model_in) = 0;
    };
}

#endif //EYE_BLINK_DETECTOR_FEATURE_EXTRACTOR_H
