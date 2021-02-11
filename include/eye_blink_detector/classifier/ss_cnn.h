#ifndef EYE_BLINK_DETECTOR_SS_CNN_H
#define EYE_BLINK_DETECTOR_SS_CNN_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "eye_blink_detector/classifier/sequence_classifier.h"

namespace eb_detector {
    class SingleShotCnn : public SequenceClassifier{
    public:
        SingleShotCnn(const YAML::Node& params);

        std::vector<float> forward(const std::vector<cv::Mat>& images) override;

        void reset() override;
    private:
        // Params
        std::string cnn_xml_;
        std::string cnn_bin_;
        int dnn_backend_;
        int cnn_input_height_, cnn_input_width_;
        bool convert_input_to_gray_;

        // Neural Network
        cv::dnn::Net cnn_;
    };
}

#endif //EYE_BLINK_DETECTOR_SS_CNN_H
