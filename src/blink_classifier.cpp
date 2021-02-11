#include "eye_blink_detector/blink_classifier.h"
#include "eye_blink_detector/eyes_cropper/eyes_band_cropper.h"
#include "eye_blink_detector/eyes_cropper/separate_eyes_cropper.h"
#include "eye_blink_detector/classifier/nn_classifier.h"
#include "eye_blink_detector/classifier/semantic_classifier.h"
#include "eye_blink_detector/utils.h"

namespace eb_detector{
    BlinkClassifier::BlinkClassifier(const YAML::Node& params){
        n_blinks = 0;

        if (!params["eyes_cropper"]) std::cerr << "Eyes Cropper params missing" << std::endl;
        if (!params["sequence_classifier"]) std::cerr << "SequenceClassifier params missing" << std::endl;


        // Get params
        getParam(params["eyes_cropper"], "type", eyes_cropper_type_, std::string("separate_eyes_cropper"));
        getParam(params["feature_extractor"], "type", feature_extractor_type_, std::string("segmentation_extractor"));
        getParam(params["sequence_classifier"], "type", sequence_classifier_type_, std::string("nn_classifier"));

        // Init Eyes Cropper
        if (eyes_cropper_type_ == "separate_eyes_cropper")
            image_cropper_ = new SeparateEyesCropper(params["eyes_cropper"]);
        else if (eyes_cropper_type_ == "eyes_band_cropper")
            image_cropper_ = new EyesBandCropper(params["eyes_cropper"]);
        else std::cerr << "Eyes Cropper type not available" << std::endl;

        // Init Feature Extractor
        if (feature_extractor_type_ == "nn_feature_extractor")
            feature_extractor_ = new NNFeatureExtractor(params["feature_extractor"]);
        else if (feature_extractor_type_ == "segmentation_extractor")
            feature_extractor_ = new SegmentationExtractor(params["feature_extractor"]);
        else std::cerr << "Feature Extractor type not available" << std::endl;

        // Init SequenceClassifier
        if (sequence_classifier_type_ == "nn_classifier")
            sequence_classifier_ = new NNClassifier(params["sequence_classifier"]);
        else if (sequence_classifier_type_ == "semantic_classifier")
            sequence_classifier_ = new SemanticClassifier(params["sequence_classifier"]);
        else std::cerr << "Sequence Classifier type not available" << std::endl;
    }

    bool BlinkClassifier::predict(const cv::Mat& image, const Face& face){
        // Crop
        std::vector<cv::Mat> imgs;
        if (image_cropper_->crop(image, face, &imgs)) {
            // Extract features
            cv::Mat features = feature_extractor_->forward(imgs);

            // Classify
            if (sequence_classifier_->forward(features)) {
                n_blinks++;
                return true;
            } else {
                return false;
            }
        } else {
            sequence_classifier_->reset();
            return false;
        }
    }

    /*void BlinkClassifier::storePrediction(){
        predictions_[0].push_back(blink_conf_[0]);
        predictions_[1].push_back(blink_conf_[1]);

        if (predictions_[0].size()>120){
            predictions_[0].erase(predictions_[0].begin());
        }
        if (predictions_[1].size()>120){
            predictions_[1].erase(predictions_[1].begin());
        }
    }*/

    /*void plotLine(cv::Mat* image, std::vector<float> points, int scale, cv::Scalar color, int thickness){
        int dt = 3;
        int y0 = 300;

        for (int i=1; i<points.size(); i++){
            cv::Point p1((i-1)*dt,y0 - points[i-1]*scale), p2(i*dt,y0 - points[i]*scale);
            cv::line(*image, p1, p2, color, thickness);
        }
    }*/

    /*void BlinkClassifier::visualizePlot(cv::Mat* image){
        plotLine(image, predictions_[0], 200, cv::Scalar(0,255,0), 2);
        plotLine(image, predictions_[1], 200, cv::Scalar(0,0,255), 2);
    }*/

    void BlinkClassifier::visualizeResults(cv::Mat* image){
        image_cropper_->visualizeResults(image);

        cv::putText(*image, "N. Blinks: " + std::to_string(n_blinks), cv::Point(20, 70),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);

        //visualizePlot(image);
    }

    void BlinkClassifier::reset(){
        sequence_classifier_->reset();
    }
}