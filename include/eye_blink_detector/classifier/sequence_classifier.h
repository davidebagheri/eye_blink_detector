#ifndef EYE_BLINK_DETECTOR_SEQUENCE_CLASSIFIER_H
#define EYE_BLINK_DETECTOR_SEQUENCE_CLASSIFIER_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "yaml-cpp/yaml.h"
#include "eye_blink_detector/utils.h"

namespace eb_detector{
    class SequenceClassifier{
    public:
        SequenceClassifier(const YAML::Node& params);

        virtual bool forward(const cv::Mat& features) = 0;

        void reset();

    protected:
        // Methods for sequence handling
        void insertToPosSequence(const cv::Mat& feat_vector, const int& id);
        void shiftSeqBackwards();
        bool addToSequence(const cv::Mat& feat_vector);

        // Params
        int sequence_len_, feature_len_, batch_size_;

        // Variables
        cv::Mat seq_blob_;
        int next_pos_;      // Position iin which the next feature vector must be added
    };
}
#endif //EYE_BLINK_DETECTOR_SEQUENCE_CLASSIFIER_H
