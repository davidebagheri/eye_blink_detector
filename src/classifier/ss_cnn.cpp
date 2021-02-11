#include "eye_blink_detector/classifier/ss_cnn.h"

namespace eb_detector{
    SingleShotCnn::SingleShotCnn(const YAML::Node& params) : SequenceClassifier(params){
        // Read CNN model
        getParam(params, "cnn_xml_path", cnn_xml_, std::string("../models/eye_cnn/eye_cnn.xml"));
        getParam(params, "cnn_bin_path", cnn_bin_, std::string("../models/eye_cnn/eye_cnn.bin"));
        getParam<int>(params, "dnn_backend", dnn_backend_, cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        getParam(params, "cnn_input_height", cnn_input_height_, 32);
        getParam(params, "cnn_input_width", cnn_input_width_, 26);
        getParam(params, "convert_input_to_gray", convert_input_to_gray_, true);

        cnn_ = cv::dnn::readNet(cnn_xml_, cnn_bin_);
        cnn_.setPreferableBackend(dnn_backend_);
    }

    std::vector<float> SingleShotCnn::forward(const std::vector<cv::Mat>& images){
        std::vector<float> result;

        for (const cv::Mat& image : images){
            // Convert to gray
            cv::Mat cnn_input = image.clone();
            if (convert_input_to_gray_)
                cv::cvtColor(image, cnn_input, cv::COLOR_BGR2GRAY);

            // Resize
            cv::resize(cnn_input, cnn_input, cv::Size(cnn_input_width_, cnn_input_height_));

            // Normalize
            cnn_input.convertTo(cnn_input, CV_32F);
            cnn_input = cnn_input/255;

            // Create blob
            cv::Mat blob;

            int sz[] = { 1, 1, cnn_input.rows, cnn_input.cols };
            blob.create(4, sz, cnn_input.depth());

            cnn_input.copyTo(cv::Mat(cnn_input.rows, cnn_input.cols, cnn_input.depth(), blob.ptr(0, 0)));

            // Forward
            cnn_.setInput(blob);
            cv::Mat confidence = cnn_.forward();
            result.push_back(confidence.at<float>(0,0));
        }

        return result;
    }

    void SingleShotCnn::reset(){}

}
