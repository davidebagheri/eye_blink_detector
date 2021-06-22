#include "eye_blink_detector/classifier/nn_classifier.h"

namespace eb_detector{

    NNClassifier::NNClassifier(const YAML::Node& params) : SequenceClassifier(params){
        getParam(params, "xml_path", xml_path_, std::string("../models/seg_lstm/seg_lstm.xml"));
        getParam(params, "bin_path", bin_path_, std::string("../models/seg_lstm/seg_lstm.bin"));
        getParam<int>(params, "dnn_backend", dnn_backend_, cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        getParam<float>(params, "confidence_threshold", confidence_th_, 0.95);
        getParam(params, "blink_class_id", blink_class_id_, 0);

        // Load model
        model_ = cv::dnn::readNet(xml_path_, bin_path_);
        model_.setPreferableBackend(dnn_backend_);

        // Normalize the confidence threshold
        confidence_th_ /= 100;
    }

    bool NNClassifier::forward(const cv::Mat& features){
        bool result = false;

        // Check if there are enough inputs in the sequence
        if (addToSequence(features)){
            // Predict
            model_.setInput(seq_blob_);
            cv::Mat model_out = model_.forward();

            // Get minimum blink confidence prediction
            float min_conf = 1;
            for (int i = 0; i < batch_size_; i++){
                min_conf = std::min(min_conf, model_out.at<float>(i, blink_class_id_));
            }

            // Apply threshold
            result = min_conf >= confidence_th_ ;
            // Clear input sequence to avoid multiple detections of very similar windows
            if (result) reset();
        }

        return result;
    }
}