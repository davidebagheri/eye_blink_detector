#ifndef EYE_BLINK_DETECTOR_NN_FEATURE_EXTRACTOR_H
#define EYE_BLINK_DETECTOR_NN_FEATURE_EXTRACTOR_H

#include "eye_blink_detector/feature_extractor/feature_extractor.h"

namespace eb_detector {
    class NNFeatureExtractor : public FeatureExtractor{
    public:
        NNFeatureExtractor(const YAML::Node& params);

        virtual cv::Mat forward(const std::vector<cv::Mat>& model_in) override;

        virtual void visualizeResults(cv::Mat* image) override {};

    protected:
        // Model paths
        std::string xml_path_;
        std::string bin_path_;

        // Model input params
        int input_width_, input_height_;
        double scale_factor_;
        bool convert_to_gray_;

        // Model
        int dnn_backend_;
        cv::dnn::Net model_;

    };
}

#endif //EYE_BLINK_DETECTOR_NN_FEATURE_EXTRACTOR_H
