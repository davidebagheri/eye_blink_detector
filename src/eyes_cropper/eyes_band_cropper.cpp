#include "eye_blink_detector/eyes_cropper/eyes_band_cropper.h"

namespace eb_detector{
    EyesBandCropper::EyesBandCropper(const YAML::Node& params) : ImageCropper(params){
        getParam(params, "x_crop_proportion", x_crop_proportion_, 0.1f);
        getParam(params, "y_crop_proportion", y_crop_proportion_, 0.1f);

        this->x_crop_proportion_ = std::min(x_crop_proportion_, 1.0f);
        this->y_crop_proportion_ = std::min(y_crop_proportion_, 1.0f);
    }

    bool EyesBandCropper::crop(const cv::Mat& image, const Face& face, std::vector<cv::Mat>* imgs){
        // Clear previous results
        imgs->clear();
        imgs_valid_ = true;

        // Compute crop offsets
        cv::Point offset = cv::Point(face.bounding_box.width * this->x_crop_proportion_,
                                     face.bounding_box.height * this->y_crop_proportion_);

        // Compute eyes ROI
        eyes_ROI_ = cv::Rect(face.eyes[0] - offset, face.eyes[1] + offset);

        // Check validity
        if (eyes_ROI_.tl().x < 0 or eyes_ROI_.tl().y < 0 or
                eyes_ROI_.br().x > image.cols - 1 or eyes_ROI_.br().y > image.rows - 1) {
            imgs_valid_ = false;
        } else {
            cv::Mat roi = image(image(eyes_ROI_));
            roi.convertTo(roi, CV_32F);
            imgs->push_back(image(roi));
        }

        return imgs_valid_;
    }

    const std::vector<cv::Rect> EyesBandCropper::getROI() const {
        return std::vector<cv::Rect>({eyes_ROI_});
    }

    void EyesBandCropper::visualizeResults(cv::Mat* image){
        cv::Scalar color;
        if (imgs_valid_){
            color = cv::Scalar(0, 0, 255);
        } else {
            color = cv::Scalar(0, 255, 255);
        }
        rectangle(*image, eyes_ROI_, color, 2);
    }
}