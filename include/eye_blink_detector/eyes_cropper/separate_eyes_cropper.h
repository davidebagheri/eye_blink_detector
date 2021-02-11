#ifndef EYE_BLINK_DETECTOR_SEPARATE_EYES_CROPPER_H
#define EYE_BLINK_DETECTOR_SEPARATE_EYES_CROPPER_H

#include "image_cropper.h"

namespace eb_detector{
    class SeparateEyesCropper  : public ImageCropper{
    public:
        SeparateEyesCropper(const YAML::Node& params);

        bool crop(const cv::Mat& image, const Face& face, std::vector<cv::Mat>* imgs) override;

        const std::vector<cv::Rect> getROI() const override;

        void visualizeResults(cv::Mat* image);

    protected:
        int eyesDistance(const Face& face);

        // crop size = distance between eyes * (x_crop_ratio_, y_crop_ratio_)
        float x_crop_ratio_;
        float y_crop_ratio_;
        bool use_manhattan_;

        std::vector<cv::Rect> eye_ROIs_;
    };
}


#endif //EYE_BLINK_DETECTOR_SEPARATE_EYES_CROPPER_H
