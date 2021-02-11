#ifndef EYE_BLINK_DETECTOR_IMAGE_CROPPER_H
#define EYE_BLINK_DETECTOR_IMAGE_CROPPER_H
#include "vector"
#include <opencv2/opencv.hpp>

#include "eye_blink_detector/face.h"
#include "eye_blink_detector/utils.h"

namespace eb_detector{
    class ImageCropper{
    public:
        ImageCropper(const YAML::Node& params){}

        virtual bool crop(const cv::Mat& img, const Face& face, std::vector<cv::Mat>* imgs)=0;

        virtual const std::vector<cv::Rect> getROI() const =0;

        virtual void visualizeResults(cv::Mat* img)=0;

    protected:
        bool imgs_valid_;   // False if the cropped images are not valid
    };
}
#endif //EYE_BLINK_DETECTOR_IMAGE_CROPPER_H
