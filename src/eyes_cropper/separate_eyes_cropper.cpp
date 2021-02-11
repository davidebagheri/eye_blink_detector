#include "eye_blink_detector/eyes_cropper/separate_eyes_cropper.h"

namespace eb_detector{
    SeparateEyesCropper::SeparateEyesCropper(const YAML::Node& params) : ImageCropper(params){
        // Get params
        getParam(params, "x_crop_ratio", x_crop_ratio_, 0.7f);
        getParam(params, "y_crop_ratio", y_crop_ratio_, 0.5f);
        getParam(params, "use_manhattan_distance", use_manhattan_, true);

        // Check max ratio values
        x_crop_ratio_ = std::min(x_crop_ratio_, 1.0f);
        y_crop_ratio_ = std::min(y_crop_ratio_, 1.0f);
    }

    bool SeparateEyesCropper::crop(const cv::Mat& image, const Face& face, std::vector<cv::Mat>* imgs){
        // Clear previous results
        eye_ROIs_.clear();
        imgs->clear();
        imgs_valid_ = true;

        for (const cv::Point& eye : face.eyes){
            // Get crop dimensions
            int eyes_distance = eyesDistance(face);
            int x_crop_size = eyes_distance * x_crop_ratio_;
            int y_crop_size = eyes_distance * y_crop_ratio_;

            // Check dimensions
            if (x_crop_size < 2 or y_crop_size < 2) return false;

            // Compute bounding box
            cv::Point upper_left = eye - cv::Point(x_crop_size/2, y_crop_size/2);
            cv::Point bottom_right = eye + cv::Point(x_crop_size/2, y_crop_size/2);
            cv::Rect eye_ROI = cv::Rect(upper_left, bottom_right);
            eye_ROIs_.push_back(eye_ROI);

            // Check validity
            if (upper_left.x < 0 or upper_left.y < 0 or
                    bottom_right.x > image.cols - 1 or bottom_right.y > image.rows - 1) {
                imgs_valid_ = false;
            } else {
                // Crop
                cv::Mat roi = image(eye_ROI);
                roi.convertTo(roi, CV_32F);
                imgs->push_back(roi);
            }
        }
        return imgs_valid_;
    }

    int SeparateEyesCropper::eyesDistance(const Face& face){
        cv::Point diff = face.eyes[0] - face.eyes[1];
        if (use_manhattan_){
            return std::abs(diff.x) + std::abs(diff.y);
        } else {
            return cv::norm(diff);
        }
    }

    const std::vector<cv::Rect> SeparateEyesCropper::getROI() const{
        return eye_ROIs_;
    }

    void SeparateEyesCropper::visualizeResults(cv::Mat* image){
        cv::Scalar color;
        if (imgs_valid_){
            color = cv::Scalar (0, 0, 255);
        } else {
            color = cv::Scalar (0, 255, 255);
        }

        for (const cv::Rect& eye_roi : eye_ROIs_) {
            rectangle(*image, eye_roi, color, 2);
        }
    }
}