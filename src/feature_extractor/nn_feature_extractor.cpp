#include "eye_blink_detector/feature_extractor/nn_feature_extractor.h"

namespace eb_detector {
    NNFeatureExtractor::NNFeatureExtractor(const YAML::Node &params) : FeatureExtractor(params) {
        // Params
        getParam(params, "xml_path", xml_path_, std::string("../models/eye_unet/FP32/eye_unet_32.xml"));
        getParam(params, "bin_path", bin_path_, std::string("../models/eye_unet/FP32/eye_unet_32.bin"));
        getParam<int>(params, "dnn_backend", dnn_backend_, 0);
        getParam<int>(params, "preferable_target", preferable_target_, 0);
        getParam(params, "input_width", input_width_, 80);
        getParam(params, "input_height", input_height_, 40);
        getParam(params, "input_scale_factor", scale_factor_, 1.0);
        getParam(params, "convert_to_gray", convert_to_gray_, false);

        // Load model
        model_ = cv::dnn::readNet(xml_path_, bin_path_);
        model_.setPreferableBackend(dnn_backend_);
    }

    cv::Mat NNFeatureExtractor::forward(const std::vector<cv::Mat>& model_in) {
        // Gray
        if (convert_to_gray_) {
            for (int i = 0; i < model_in.size(); i++) {
                cv::cvtColor(model_in[i], model_in[i], cv::COLOR_BGR2GRAY);
            }
        }

        cv::Mat blob = cv::dnn::blobFromImages(model_in, scale_factor_, cv::Size(input_width_, input_height_));
        model_.setInput(blob);

        return model_.forward();
    }
}