#ifndef EYE_BLINK_DETECTOR_CNN_LSTM_H
#define EYE_BLINK_DETECTOR_CNN_LSTM_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "eye_blink_detector/classifier/sequence_classifier.h"

namespace eb_detector {
    class CnnLstm : public SequenceClassifier{
    public:
        CnnLstm(const YAML::Node& params);

        std::vector<float> forward(const std::vector<cv::Mat>& images) override;

        void reset() override;

    private:
        void insertToPosSequence(const cv::Mat& feat_vector, const int& id);

        void shiftSeqBackwards();

        // Params
        std::string cnn_xml_;
        std::string cnn_bin_;
        std::string lstm_xml_;
        std::string lstm_bin_;
        int dnn_backend_;
        int cnn_input_height_, cnn_input_width_;
        int sequence_lenght_;
        int img_feature_lenght_;
        int batch_size_;

        // Variables
        cv::Mat last_img_feat_; // Feature map from last image
        cv::Mat seq_blob_;      // LSTM input
        int n_imgs_in_seq_;     // number of images considered in the sequence

        // Models
        cv::dnn::Net cnn_;      // Feature extractor CNN
        cv::dnn::Net lstm_;     // Feature sequence classifier
    };
}


#endif //EYE_BLINK_DETECTOR_CNN_LSTM_H
