#include "eye_blink_detector/classifier/cnn_lstm.h"

namespace eb_detector{
    CnnLstm::CnnLstm(const YAML::Node& params) : SequenceClassifier(params){
        // Get params
        getParam(params, "cnn_xml_path", cnn_xml_, std::string("../models/cnn/cnn_16_2.xml"));
        getParam(params, "cnn_bin_path", cnn_bin_, std::string("../models/cnn/cnn_16_2.bin"));
        getParam(params, "lstm_xml_path", lstm_xml_, std::string("../models/lstm/lstm_16_2.xml"));
        getParam(params, "lstm_bin_path", lstm_bin_, std::string("../models/lstm/lstm_16_2.bin"));
        getParam(params, "cnn_input_height", cnn_input_height_, 50);
        getParam(params, "cnn_input_width", cnn_input_width_, 50);
        getParam(params, "sequence_lenght", sequence_lenght_, 10);
        getParam(params, "img_feature_lenght", img_feature_lenght_, 576);
        getParam(params, "batch_size", batch_size_, 2);
        getParam<int>(params, "dnn_backend", dnn_backend_, cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);

        // Init the neural networks
        cnn_ = cv::dnn::readNet(cnn_xml_, cnn_bin_);
        cnn_.setPreferableBackend(dnn_backend_);
        lstm_ = cv::dnn::readNet(lstm_xml_, lstm_bin_);
        lstm_.setPreferableBackend(dnn_backend_);

        // Init varibales
        std::vector<int> lstm_input_shape = std::vector<int>({batch_size_, sequence_lenght_-1, img_feature_lenght_});
        seq_blob_ = cv::Mat(lstm_input_shape, CV_32F);
        last_img_feat_ = cv::Mat(std::vector<int>({batch_size_, 1, img_feature_lenght_}), CV_32F);
        n_imgs_in_seq_ = 0;
    }

    void CnnLstm::insertToPosSequence(const cv::Mat& feat_vector, const int& id){
        cv::Range ranges[3];
        ranges[0] = cv::Range::all();
        ranges[1] = cv::Range(id,id + 1);
        ranges[2] = cv::Range::all();
        feat_vector.copyTo(seq_blob_(ranges));
    }

    void CnnLstm::shiftSeqBackwards(){
        for (int i=1; i<sequence_lenght_-1; i++){
            cv::Range ranges[3];
            ranges[0] = cv::Range::all();
            ranges[1] = cv::Range(i,i + 1);
            ranges[2] = cv::Range::all();

            insertToPosSequence(seq_blob_(ranges), i - 1);
        }
    }

    std::vector<float> CnnLstm::forward(const std::vector<cv::Mat>& images){
        if (images.size() != 2){
            reset();
            return std::vector<float>({-1, -1});
        }

        // Compute image feature with CNN
        cv::Mat blob = cv::dnn::blobFromImages(images, 1, cv::Size(cnn_input_width_, cnn_input_height_));
        cnn_.setInput(blob);
        cv::Mat curr_img_feat = cnn_.forward();

        // Reshape to fit LSTM input and store
        curr_img_feat = curr_img_feat.reshape(1, std::vector<int>({2, 1, curr_img_feat.size[1]}));

        // First image case
        if (n_imgs_in_seq_ == 0){
            curr_img_feat.copyTo(last_img_feat_);
            n_imgs_in_seq_++;
            return std::vector<float>({-1, -1});
        }

        // Compute difference with previous feature map
        cv::Mat feat_diff = curr_img_feat - last_img_feat_;
        curr_img_feat.copyTo(last_img_feat_);

        // Not enough images to compute classification
        if (n_imgs_in_seq_ < seq_blob_.size[1] + 1){
           // Add difference to the sequence
            insertToPosSequence(feat_diff, n_imgs_in_seq_ - 1);
           n_imgs_in_seq_++;
           if (n_imgs_in_seq_ == seq_blob_.size[1] + 1){
               lstm_.setInput(seq_blob_);

               cv::Mat res = lstm_.forward();
               std::vector<float> eye_blink_conf = std::vector<float>({
                   res.at<float>(0, 1),
                   res.at<float>(1, 1)});
               return  eye_blink_conf;
           }

           return std::vector<float>({0, 0});
        }

        // Shift feature difference temporal window
        shiftSeqBackwards();

        // Add last input
        insertToPosSequence(feat_diff, n_imgs_in_seq_ - 2);

        // Compute the result
        lstm_.setInput(seq_blob_);
        cv::Mat res = lstm_.forward();

        std::vector<float> eye_blink_conf = std::vector<float>({
            res.at<float>(0, 1),
            res.at<float>(1, 1)});

        return  eye_blink_conf;
    }

    void CnnLstm::reset(){
        n_imgs_in_seq_ = 0;
    }
}