#ifndef EYE_BLINK_DETECTOR_EYES_BAND_CROPPER_H
#define EYE_BLINK_DETECTOR_EYES_BAND_CROPPER_H

#include "image_cropper.h"

namespace eb_detector{
    class EyesBandCropper : public ImageCropper{
    public:
        EyesBandCropper(const YAML::Node& params);

        bool crop(const cv::Mat& image, const Face& face, std::vector<cv::Mat>* imgs) override;

        const std::vector<cv::Rect> getROI() const override;

        void visualizeResults(cv::Mat* image);

    protected:
        // proportions of width and height face taken as offset for eye region crop
        float x_crop_proportion_, y_crop_proportion_;

        // ROI containing both eyes
        cv::Rect eyes_ROI_;
    };
}



#endif //EYE_BLINK_DETECTOR_EYES_BAND_CROPPER_H
