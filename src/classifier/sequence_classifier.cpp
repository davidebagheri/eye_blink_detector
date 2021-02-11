#include "eye_blink_detector/classifier/sequence_classifier.h"

namespace eb_detector{

    SequenceClassifier::SequenceClassifier(const YAML::Node& params){
        getParam(params, "batch_size", batch_size_, 2);
        getParam(params, "sequence_length", sequence_len_, 10);
        getParam(params, "img_feature_length", feature_len_, 576);

        // Initialize variables
        next_pos_ = 0;
        std::vector<int> lstm_input_shape = std::vector<int>({batch_size_, sequence_len_, feature_len_});
        seq_blob_ = cv::Mat(lstm_input_shape, CV_32F);
    }

    void SequenceClassifier::reset(){
        next_pos_ = 0;
    }

    void SequenceClassifier::insertToPosSequence(const cv::Mat& feat_vector, const int& id){
        cv::Range ranges[3];
        ranges[0] = cv::Range::all();
        ranges[1] = cv::Range(id,id + 1);
        ranges[2] = cv::Range::all();
        feat_vector.copyTo(seq_blob_(ranges));
    }

    void SequenceClassifier::shiftSeqBackwards(){
        for (int i=1; i < sequence_len_; i++){
            cv::Range ranges[3];
            ranges[0] = cv::Range::all();
            ranges[1] = cv::Range(i,i + 1);
            ranges[2] = cv::Range::all();

            insertToPosSequence(seq_blob_(ranges), i - 1);
        }
    }

    bool SequenceClassifier::addToSequence(const cv::Mat& feat_vector){
        if (next_pos_ < sequence_len_-1){
            insertToPosSequence(feat_vector, next_pos_);
            next_pos_++;
            return 0;
        }

        shiftSeqBackwards();
        insertToPosSequence(feat_vector, next_pos_);
        return 1;
    }

}