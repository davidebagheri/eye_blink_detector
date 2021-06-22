#ifndef EYE_BLINK_DETECTOR_NN_CLASSIFIER_H
#define EYE_BLINK_DETECTOR_NN_CLASSIFIER_H

#include <opencv2/dnn.hpp>

#include "eye_blink_detector/classifier/sequence_classifier.h"

namespace eb_detector {
    class NNClassifier : public SequenceClassifier {
    public:
        NNClassifier(const YAML::Node& params);

        virtual bool forward(const cv::Mat& features) override;

    protected:
        // Params
        float confidence_th_;
        int blink_class_id_;

        // Model paths
        std::string xml_path_;
        std::string bin_path_;
        int dnn_backend_;
        int preferable_target_;

        // Model
        cv::dnn::Net model_;
    };
}

#endif //EYE_BLINK_DETECTOR_NN_CLASSIFIER_H
