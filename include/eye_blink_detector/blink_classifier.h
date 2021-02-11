#ifndef EYE_BLINK_DETECTOR_BLINK_CLASSIFIER_H
#define EYE_BLINK_DETECTOR_BLINK_CLASSIFIER_H

#include <opencv2/opencv.hpp>

#include "eye_blink_detector/face.h"
#include "eye_blink_detector/eyes_cropper/image_cropper.h"
#include "eye_blink_detector/classifier/sequence_classifier.h"
#include "eye_blink_detector/feature_extractor/segmentation_extractor.h"
#include "eye_blink_detector/feature_extractor/nn_feature_extractor.h"
#include "yaml-cpp/yaml.h"

namespace eb_detector {
    class BlinkClassifier {
    public:
        BlinkClassifier(const YAML::Node& params);

        bool predict(const cv::Mat& image, const Face& face);

        void visualizeResults(cv::Mat* image);

        void reset();

        //void visualizePlot(cv::Mat* image);

    private:
        //void storePrediction();

        // Params
        std::string eyes_cropper_type_;
        std::string sequence_classifier_type_;
        std::string feature_extractor_type_;

        // Variables
        std::vector<float> blink_conf_; // Current detection blink confidence
        int n_blinks;                    // number of predicted blinks

        std::vector<float> predictions_[2];   // Current and past predictions to plot

        // Eye cropper
        ImageCropper* image_cropper_;

        // Feature extractor
        FeatureExtractor* feature_extractor_;

        // SequenceClassifier
        SequenceClassifier* sequence_classifier_;
    };
}

#endif //EYE_BLINK_DETECTOR_BLINK_CLASSIFIER_H
