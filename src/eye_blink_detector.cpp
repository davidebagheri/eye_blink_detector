#include "eye_blink_detector/eye_blink_detector.h"

namespace eb_detector{

    EyeBlinkDetector::EyeBlinkDetector(const YAML::Node& params) {
        if (!params["face_detector"]) std::cerr << "Face Detector params missing" << std::endl;
        if (!params["blink_classifier"]) std::cerr << "Blink SequenceClassifier params missing" << std::endl;

        face_detector_ = new FaceDetector(params["face_detector"]);
        blink_classifier_ = new BlinkClassifier(params["blink_classifier"]);
    }

    bool EyeBlinkDetector::detect(const cv::Mat& image){
        face_detector_->detectFaces(image);

        if (face_detector_->selectFace(image)) {
            blink_classifier_->predict(image, face_detector_->getSelectedFace());
            return true;
        } else {
            blink_classifier_->reset();
        }
        return false;
    }

    void EyeBlinkDetector::visualizeResults(cv::Mat* image){
        face_detector_->visualizeResults(image);
        blink_classifier_->visualizeResults(image);
    }

    const BlinkClassifier& EyeBlinkDetector::getBlinkClassifier(){
        return *blink_classifier_;
    }

    const FaceDetector& EyeBlinkDetector::getFaceDetector() {
        return *face_detector_;
    }
}